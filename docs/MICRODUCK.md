# MicroDuck mobile runtime

MicroDuck is an opt-in mobile base with a bounded velocity interface. It does
not instantiate an arm, inverse kinematics, grasping, or the kitchen scene.
The implementation is a candidate: general walking and turning have not passed
physical admission. The optional [geometric distance candidate](MICRODUCK_DISTANCE_CANDIDATE.md)
retains four historically confirmed native fresh-start ±30 mm receipts. A later
[observation audit](evidence/voice-reply-recovery-20261004/guarded-continuation.json)
finds heading drift after completion in both forward cases, exceeding the
unchanged 0.08 rad bound; those receipts do not admit the current contract.
Separate evidence covers a stop during movement and one small negative turn;
composed reverse motion and larger or positive turns still have retained failures.
PhysX BAM, scene reset, hardware, and hosted conversation
are not delivered by this locomotion change.

The implementation continues the [approved design](superpowers/plans/2026-10-01-microduck.md)
and the [dated review](MICRODUCK_DESIGN_REVIEW_20261002.md). That review describes
an earlier source snapshot; the current continuation adds inference cancellation
fencing, explicit solved support, and complete effective-model identity.
The [continuation report](MICRODUCK_CONTINUATION_20261002.md) preserves its
original native MCP outcomes and failures. The [current capability index](PROJECT_STATUS_20261003.md)
links later distance and conversation results, with their source/model limits.
The [modular architecture](ROBOT_MODULARITY.md) separates single-robot control
from shared-scene ownership; the ordinary native bridge still owns one robot.

## Runtime and clocks

`MobileBase` supplies immutable physical state and bounded body-frame velocity
commands. `SafeBase` enforces limits, freshness, cancellation and progress.
`MobileRig` selects bases by name. `MobileSkillRuntime` records each execution
and its independently sampled postcondition in the normal CASCADE trace and
episodic memory. The MCP reader handles stop and cancellation while the worker
is inside a motion call.

The native bridge runs a floating-root articulation in a separate Isaac Sim
arena. The current executable backend is Newton / MJWarp, using the
pinned native BAM implementation. The official `velstand.onnx` consumes 61
observations and emits 14 actions. Physics advances nominally every 5 ms and
the policy runs every four completed solves; the actual native timestep is
recorded. Camera capture and LLM latency do not advance the policy clock.
Neck and head joints belong to the policy; another controller must not write
them. The mouth is outside its action vector.

Inference previews do not mutate policy history. The bridge rechecks command,
generation, episode, physical clocks and wall deadlines before committing an
inference result. Stop invalidates walking intent while allowing the policy
and actuators to maintain balance. It does not certify an instantaneous
physical stop or promise to undo a motor target already committed.

The policy API and single/shared launchers offer the explicit
`target_profile="robotd-targets-v1"` / `--target-profile robotd-targets-v1`
candidate. It reproduces the gait output transform in
[robotd at 9136aa4](https://github.com/pollen-robotics/microduck/blob/9136aa4ee88e81edf2bcaf3527e90b65da25f1eb/robotd/src/control.rs#L607):
float64 HOME plus `0.9 * raw_action`, then target EMA with alpha 0.5 for the
four head joints and 0.7 for the ten leg joints. The first target after
construction/reset is unfiltered; subsequent targets use the last committed
filtered target. Raw float32 actions remain the observation history. Preview
and discarded evaluations cannot move either history. After `infer()` or
`commit()`, `committed_targets` returns the accepted target; `targets(action)`
always previews the next slot. `target_contract` binds the profile and policy
digest. The default `direct-v1` retains scale 1.0 and no filtering.

This is a target-transform candidate, not the complete robotd pipeline or its
separate standing/skill overrides. It does not add command smoothing, voltage
adaptation, stale-sensor coasting or actuator delay. Launchers keep `direct-v1`
as the default and reject a missing, changed or silently ignored target contract.
The effective model binds this selection separately from the policy weights.
Its `target_upload` record distinguishes calculation precision from the existing
BAM upload: filtered targets retain float64 history, then `set_targets` converts
them to float32 before Warp assignment. No native result is transferred between
profiles. The current official VelStand LFS digest
is the same `1c659be5…` used by the retained eight-case negative corpus; its
published provenance names `protective_fall` and a September 14 export but
does not identify the exact training run. CPU formula replay cannot predict
the new closed-loop actions or establish improved gait or stopping.

The [native target-transform foundation](evidence/robot-modularity/microduck-robotd-target-foundation-20261004.json)
retains one robot at source `2c27f846`, with 800 completed solves, 200 policy
commits, nine camera/support samples and two reader events. The independent
auditor reproduced every raw-action history and float64 target, then checked
all 800 float32 BAM uploads. No preview was discarded. Maximum displacement
from the first solve was 2.6421 mm, maximum tilt 0.0206804 rad, and minimum
sole support over the final two simulated seconds was 7.2072 N. These zero
caller commands still caused real postural actuation; they do not admit walking,
turning, braking or a verified stop. Both weights and target transformation
differ from the prior rough/direct reader. Native and owned scope exited zero,
all seven observed process births disappeared, and inputs and protected stores
were unchanged. Owner-thread attempt duration was 11.580 ms median and 109.550 ms maximum;
no generation-2 GC occurred, so this is neither a sustained deadline guarantee
nor a causal speed comparison under the recorded foreign GPU occupancy.
The separate [Newton BAM proposal](https://github.com/newton-physics/newton/issues/4397)
motivates reviewing actuator contracts; this episode retains its pinned BAM
plant and does not test a new friction, backlash or delay implementation.

The subsequent [single-robot forward attempts](evidence/robot-modularity/microduck-robotd-forward-failures-20261004.json)
retain two failures with this same source and model. The first omitted the
composed-runtime tag and failed its initial MCP observation before reset or
motion. After a passive composition regression and a separately reviewed
harness correction, the second reached the real `walk_distance(0.025)` path.
It reported 10.6337 mm before its configured three-second simulation deadline,
below the original 25 mm target with 5 mm tolerance. The full-rate journal
never entered that tolerance, including the retained tail: maximum forward
travel was 10.6421 mm, with no heading/lateral or forbidden-support veto.
The client therefore withheld the scored stop/rest phase; containment ACKs
do not establish physical stopping.

Each episode retained 2,000 solves, 500 policy commits and 21 camera/support
samples, with ordinary native/scope exits and closed clients. In the second,
150 policy rows carried the observed forward command and every raw-history,
filtered-target and float32 upload was reproduced. The failure envelope and
both traces omitted the admission ACK; the original auditor failures remain
preserved. A separate supplement joins the actual command, controller and
physical records without reconstructing that missing receipt or granting
admission. Per-joint target/tracking/torque statistics are diagnostic only.
Neither episode exercised generation-2 GC, establishes causality for the
target transformation, or admits locomotion or a verified physical stop.

The separate [headless MuJoCo prefix](evidence/robot-modularity/microduck-mujoco-prefix-negative-20261004.json)
retains one negative with the same VelStand weights and target transformation.
It stopped at solve 10 (0.05 simulated seconds), after three policy commits,
before any forward command. The frozen preparation gate rejected constraint
contacts between the floor and both foot housings; each had zero normal force,
while the soles carried 23.4769 N. This gate applied throughout preparation,
whereas the historical CPU corpus checked support at its end and native
geometric control began later. The same housing pairs, also at zero normal
force, occur at the corresponding retained native step 12. This result therefore
identifies a difference in preparation criteria, not a demonstrated fall or
backend cause of the earlier walking failure.

The first 61-element observation, raw actions, float64 filtered targets and actual
float32 BAM uploads matched the native prefix exactly; the next two policy
commits differed. Source XML/meshes and primary physical fields were bound,
but compiled mesh equivalence and solver equivalence were not established.
The episode exited with code 1 and both observed process births absent,
unchanged inputs, and no signals or forced closure. The original negative and
all contacts remain retained. No active-motion, tail, rest or verified-stop
window was reached, and no retry or parameter adjustment followed.

The [bootstrap A/B diagnostic](evidence/robot-modularity/microduck-bootstrap-ab-20261004.json)
completed two solves using one pinned MuJoCo model,
two independent `MjData` states and the retained first policy target, without an
ONNX session or inference. Case A reproduced the original first actuator and
completed-state rows exactly. Case B changed only the external-force argument
of the first BAM friction calculation to zero; its original force buffers,
motor calculation, target, initial state, solver and timestep were preserved.

Within this CPU setup, that substitution changed the maximum joint velocity by
0.00537643258 rad/s and the friction budget by 0.00246892206 Nm. Against the
retained Newton first row, the maximum q/dq/friction residual fell from
2.68850674e-5 rad / 0.00537642489 rad/s / 0.00246892205 Nm in A to
1.37083754e-8 rad / 3.01011785e-8 rad/s / 1.56197433e-11 Nm in B.
The original CPU external input is now observed directly: right-hip-roll
received 0.030560578339 Nm before the substitution. This establishes the effect
of that input on this CPU first solve. The small residual against Newton does
not establish equivalence of integrators, compiled geometry or later dynamics.

The owned scope exited zero, without signals, forced termination or remaining
processes; both observed births disappeared. This diagnostic exercised no
DRIVE interval, sequential rollout, locomotion task or physical stop/rest
window. The earlier failed walk and preparation gate remain failed. No
production physics setting or locomotion controller changed.

The subsequent [effective-options capture](evidence/robot-modularity/microduck-effective-options-20261004.json)
observed `implicitfast` (3) in both the CPU MuJoCo and Warp models under the
original native recipe. Their other options differ: CPU timestep 0.002 s and
tolerance approximately 1e-8, versus Warp timestep 0.004999999888241291 s and
tolerance approximately 1e-6. Agreement applies only to the integrator. The
solver configuration had no integrator override; the imported Newton model
held 3. USD exposed `euler` as an unauthored fallback with no property stack,
which did not select the effective integrator.

Two identical snapshots retained the same seven native model/solver/state
objects and step-2 clock. The original startup performed two native bootstrap
solves, then initialized HOME/FK and bound/reset BAM; the snapshots followed
that initialization. No policy, command endpoint or episode solve ran. The
complete identity recipe matched foundation02 except for the absent episode
profiler label. Native and scope exits were zero, all eight observed births
were absent, and no signals or forced closure occurred. This measures the
current recipe, not the historical gait's effective options, and does not
establish an integrator cause, training alignment, locomotion or physical rest.

## Evidence and identity

A SHA-256 digest binds the complete admitted bundle, consumed USD layers,
normalized runtime scene, ONNX weights, native BAM source and effective
parameters, actual solver/device/timestep, runtime source, and contact
registry. A client must explicitly pin this digest in
`model_identity_sha256`; learning it from an arbitrary `hello` is not admission.
State, camera, and independent truth channels enforce the same binding.
Changing the recipe requires a new identity and physical evaluation.

Collision meshes carry an explicit convex-hull vertex limit, on two layers.
Newton's USD importer (Isaac Sim 6.2 / Newton 1.6.1rc1, which passes no
`mesh_maxhullvert`) resolves only an authored `newton:maxHullVertices`,
`mjc:maxhullvert` or `physxConvexHullCollision:hullVertexLimit`; an unauthored
mesh falls back to Newton's own 64-vertex cap, and because qhull's partial hull
depends on vertex order the mirrored soles came out 2.3 mm apart, which curled a
20 s straight command by +3.3 rad. Both admitted assets are MJCF conversions
whose meshes author no limit, which in MJCF means a complete hull (`-1`).
First layer, the converter: `scripts/convert_microduck.py` authors each
collision mesh's MJCF limit (`newton:maxHullVertices`, `-1` included) and its
validator refutes a missing, different or competing limit, so bundles from the
previous adapter must be reconverted. Second layer, the owner: before the stage
parse, `author_collision_hull_limits` gives every enabled convex-hull collision
mesh under each robot root an explicit limit in the anonymous runtime layer,
keeping and recording an authored one and authoring `-1` otherwise (the Isaac
Lab USD instances its colliders, so the enclosing instance prims are
de-instanced first; composition only, the pinned bytes never change). After
bootstrap, `read_native_collision_hulls` reads back the hull Newton actually
built for every colliding mesh shape (`shape_source.maxhullvert`, hull vertex
count) and refuses a limit that did not reach the native model. Both records
(`collision_hull_limits`, `collision_hulls`) enter the model identity. The
[converted-bundle comparison](evidence/microduck-hull-limits-20261004/live-gate.json)
shows Newton building the complete, symmetric source hulls (soles 4964/5029
vertices instead of 64) and the 20 s open-loop curl dropping from +3.5 to
+1.3..+1.7 rad; on the admitted Isaac Lab USD the
[owner-side comparison](evidence/microduck-hull-limits-20261005/live-gate-external-usd.json)
(two fresh-process replicas per arm, same policy and limits) drops the curl
from +1.27/+1.61 rad to +0.34/+0.21 rad and the lateral drift from 1.84/1.81 m
to 0.19/0.13 m over 2.3 m of forward travel. The remaining curl is larger than the official MuJoCo
reference (+0.5 rad for the same complete-hull model) and is not explained or
admitted by this fix.

Support uses versioned, complete post-solve contact records: exact shape
pairs, points, normals and reaction forces in world coordinates, bound to
the same epoch, step, time and model identity as pose. The independent checker
uses an explicit registry of sole and ground shapes. Empty contacts do not
prove standing; unavailable forces remain unknown. Stationary windows require
upward foot support and reject forbidden external body support. Walking phases
are evaluated separately from rest windows.

An emergency-stop acknowledgement always reports physical verification as
pending/unverified. A separate observer can later confirm or refute that stop,
with the same receipt ID. It cannot repair the outcome of an earlier failed
motion. Late, pending and duplicate observations cannot supply positive settling
evidence, but their faults, residual motion and support failures still veto an
earlier quiet window. See [stop verification and native finalization](MICRODUCK_STOP_VERIFICATION.md)
for this distinction, signal handling and provisional SDK-shutdown receipts.
`reset_stop` only restores command permission; it does not reset the
physics world or prove a new scene epoch.

## Running the candidate

The single and shared launchers accept `--integrator-profile euler-v1` as an
explicit experimental selection. The default, `sdk-default`, leaves solver
selection untouched. Euler authors the existing `mjc:option:integrator` token
after physics setup and before scene export and the original two bootstrap
solves. Preparation rejects an absent attribute or any disagreement between
the authored value, imported model and effective CPU/Warp integrators.
The model identity includes this contract and the observed options; a second
read after identity binding must match before policy or episode work starts.
CPU and Warp timesteps and tolerances are retained separately, not equated.
This permits a bounded comparison with the official inference script's Euler
setting. It does not establish the checkpoint's training configuration, explain
the earlier distance failure, or admit a new gait. Each selected recipe requires
its own native preparation, model pin and reviewed motion episode; gains,
cadence, bootstrap forces and physical acceptance limits are unchanged.

The [Euler preparation](evidence/robot-modularity/microduck-euler-preparation-20261005.json)
at source `f7b1bd1a` retained the two original bootstrap solves and no policy
construction, ONNX inference, episode solve or command endpoint. Those bootstrap
solves precede the existing HOME/FK writes and BAM binding/reset. The imported
model and effective CPU/Warp options reported Euler (`0`); snapshots before and
after identity binding were byte-identical. The new model digest is `a74083e7…`.
Against the retained default preparation, the auditor found only the declared
source map, authored USD Euler token and integrator identity record changed.
All other effective options matched that baseline. CPU timestep/tolerance remain
0.002 s/about 1e-8, while Warp retains float32 0.005 s/about 1e-6; agreement here
concerns the integrator only. Native and owned scope exited zero, all six observed
process births disappeared, and no signal or forced cleanup was needed. The frozen
auditor passed, and an independent rerun reproduced its result. This preparation
does not establish successful
walking, braking, physical stopping, solver equivalence or hardware operation;
the previous distance failures remain retained.

Offline smoke with the kinematic mock:

```sh
PYTHONPATH="$PWD/src" CASCADE_BASE=microduck_mock CASCADE_LLM=mock \
  python -m cascade.apps.mcp_server
```

The mock exposes the mobile tool surface and explicitly cannot verify physical
motion. Mobile tools include `get_base_state`, `walk_velocity`, `turn`,
`stop_navigation`, `emergency_stop`, `reset_stop`, and `verify_last_action`.
`camera_snapshot` is available when a camera channel is configured.

For native execution, `scripts/isaac_microduck_bridge.py --help` lists required
paths, hashes, BAM profile, private port, limits and episode deadlines.
`--check-only` verifies offline inputs without opening Kit or a socket. Use
the selected Isaac release's `python.sh`, its matching native dependencies,
a new output directory and an outer process deadline. `BRIDGE_LISTENING`
means transport availability, not locomotion acceptance.

`configs/bases/microduck_isaac.yaml` is an incomplete candidate template.
A reviewed run profile must supply actual engine/device, bundle-root and policy
hashes, the independently inspected model digest, private endpoint, explicit
support registry and verifier limits. There is no default admitted recipe.
Do not change gains, checkpoints, speed limits or verifier tolerances merely
to turn a failed campaign green.

The bridge saves startup admission, the canonical model identity, runtime
receipt, every physical step, actual policy inputs/outputs and discarded
inferences, and images with episode/step timestamps. Check process exit and
owned-resource teardown separately from the receipt. Keep failed episodes.

Isaac Sim compatibility is selected explicitly with `--sdk-recipe`; the
supported dependency versions and source checks live in
[`sim/microduck_sdk.py`](../src/cascade/sim/microduck_sdk.py). Offline checks
read files only, and initialization rechecks the modules actually imported.
The selected dependencies and solver outputs enter the effective model
identity. A supported installation does not establish balance or locomotion:
each new combination still requires its own single-robot native validation.

`--private-rtx-cache` optionally redirects only the two writable RTX shader/PSO
caches to `rtx-cache/` inside the new output directory. It requires the exact
`isaac62_48b2d951` recipe and checks the cache-routing SDK sources. The default
keeps SDK cache selection. Bundled application caches, rendering, physics and
deadlines are unchanged. Kit receives the paths before initialization; effective
settings and loaded sources are checked before the bridge can publish.

An optional `--rtx-cache-seed DIR --rtx-cache-seed-sha256 SHA` requires a separately
reviewed manifest from a successfully closed warming probe. `DIR` contains only
`manifest.json` and `data/`; the manifest binds the exact cache policy and the
complete file/directory inventory returned by
[`private_rtx_cache.inventory`](../src/cascade/sim/private_rtx_cache.py).
Its schema is `cascade.private-rtx-cache-seed.v1`, with `policy` and `inventory`
fields. Admission rejects extra, missing or changed members, symlinks and special
files. Bootstrap rechecks and copies the seed into an exclusive writable directory,
then verifies the seed and copy again. The two absolute paths and input inventory
are retained in `rtx-cache.json`; measured settings go in `rtx-cache-effective.json`.
Only the relative cache policy and implementation hashes enter model identity,
allowing independently admitted probe/episode paths to differ. Seed bytes remain
separately bound deployment inputs.

The [native cache validation](evidence/microduck-private-rtx-20261004/native.json)
retains the failed SDK-origin import and the separate one-step readiness timeout.
A renderer-only preparation subsequently closed normally in 166 seconds. Its
verified cache copy admitted source `eac66e6` to an 800-step zero-command run:
200 policy commits, 41 captures, bridge readiness at 10.7 seconds and native
duration 18.4 seconds, within the original 180/240-second limits. Both writable
destinations and loaded SDK origins matched their admissions; native and scope
exits were zero, with no shutdown signals or changed inputs. This supplies a
new source-bound model identity for the next episode. Balanced rest, commanded
locomotion and the spoken reply still require their own verification; one run
does not establish general startup reliability or a causal speed improvement.
The subsequent [VOICE04 episode](CONVERSATION.md) obtained the real walk/stop
pair but correctly refuted the walk for heading drift after completion. Its
fallback stop acknowledgement and closed resources did not establish the
requested rest-and-spoken-reply cycle.

## Isaac Lab USD folder as the admitted asset (5 October 2026)

The admitted MicroDuck asset is now the Isaac Lab MicroDuck USD folder
(`Robots/PollenRobotics/MicroDuck`, the Isaac Lab
[#8265](https://github.com/isaac-sim/IsaacLab/pull/8265)/#8266 authoring of
2 October 2026 from `microduck_rl@d424a0c8`), supplied as a plain downloaded
folder and pinned by content in
[`assets/microduck/isaaclab-microduck-usd-manifest.json`](../assets/microduck/isaaclab-microduck-usd-manifest.json)
(eight files, listing digest `0a295b65…`). `scripts/microduck_assets.py --check`
admits that folder from a local directory only; the manifest's `folder` source
kind has no download path. `scripts/admit_microduck_usd.py --build` turns it
into a schema-2 bundle (`usd/microduck_allcollisions.usd`, the folder's
`LICENSE` and `ATTRIBUTION.txt`, provenance records, `receipt.json`) and
`verify_bundle` accepts schema 2 next to the converted schema-1 bundle; the
bundle kind and variant enter the model identity. The launchers are unchanged:
`--bundle <dir> --bundle-sha256 <receipt digest>`. The USD files embed the Pollen
Robotics meshes, so they stay under the separate models notice; only the two
text files carry the Isaac Lab Apache-2.0 label.

What the owner does differently for this asset kind, all before the stage is
parsed and all recorded in the receipt: the pinned BAM sources are loaded first,
which registers `NewtonBamDriveAPI` with Newton's actuator registry; the fourteen
authored `NewtonActuator` prims are parsed through
`newton.actuators.parse_actuator_prim`, their seventeen motor/gearbox
coefficients and flags must equal the pinned Rhoban m6 fit, their deployment attributes are recorded
next to the admitted profile, and the prims are deactivated so Newton builds no
second actuator pipeline; the zero-gain `PhysicsDriveAPI:angular` on the servo
joints is removed in the runtime layer, so the joints import unactuated (target
mode NONE, effort limit 1e6) like the converted bundle; the ground is not
enrolled in any collision group because the asset authors none; the sole prims
are the instance proxies `left_foot_collision/sole_left` and
`right_foot_collision/sole_right` (`FOOT_SHAPES_BY_ASSET`); and the imported
joint properties asserted before the explicit BAM replacements are the Isaac
values (`ASSET_SOURCE_PROPERTIES`: damping 0.005359668 s·N·m/rad, armature
0.0018 kg·m², friction 0.004771183 N·m) instead of the converted bundle's
degree-based 0.053/0.0018/0.0048. The replacements themselves are unchanged:
M6 viscous damping, M6 armature and the explicit effort cap. The asset's ten
enabled convex-hull colliders author no hull vertex limit either (its root layer
was written by MuJoCo USD Converter v0.2.0), so the owner's
`author_collision_hull_limits` de-instances their ten instance prims in the
runtime layer and authors `newton:maxHullVertices = -1` on each; the receipt's
`collision_hulls` then shows the complete hulls Newton built (see *Evidence and
identity*).

Differences that change the physics, stated from the USD and MJCF sources
(CPU probes with Newton 1.6.1rc1 and Isaac Sim's `add_usd` arguments, not an
engine comparison): the Isaac all-collisions USD enables 10 convex hulls
(trunk, both hips, both shins, three head meshes, two soles) where the converted
velstand bundle enables 70 with per-geom condim/priority rules; the asset authors
the deployment distribution of Isaac Lab's training (kp 200, 7.4 V nominal, no
sag gain, current limit 1.75 A, command delay 3 to 6 steps, maximum effort
1.0676 N·m) while CASCADE keeps its admitted `official_infer_nominal_no_delay`
profile (no delay, sag gain 0.1, no current limit, 0.9634 N·m); masses, joint
limits, joint order by name, root height and the 0.96 N·m MJCF force range are
identical. The MJCF revisions differ (`d424a0c8` versus the converted bundle's
`8d0db749`). None of this is a locomotion result: the velstand checkpoint was
trained against neither of these USD files, and every status in the schema-2
receipt starts as unverified.

Newton 1.6.1rc1 is the version bundled with the admitted Isaac Sim 6.2 release.
Outside Kit it is accepted only through the explicit CPU contract recipe
`newton161rc1_cpu` (`sim/microduck_sdk.py`), which rechecks the same three
solver module digests the SDK recipe pins; stable Newton keeps the default
admission and no version range admits a pre-release. With that recipe the
pinned Isaac Lab #8161 sources (`28aa1fca`) pass `tests/test_newton_bam.py` and
`tests/test_microduck_sdk_recipe.py` on CPU (127 cases; Newton 1.6.1rc1, Warp
1.18.0, MuJoCo and MuJoCo Warp 3.12.0; recorded in `assets/microduck/newton-bam.json`).
Newton's own BAM drive ([newton-physics/newton#4504](https://github.com/newton-physics/newton/pull/4504),
towards 1.7, proposed in [#4397](https://github.com/newton-physics/newton/issues/4397))
registers `NewtonBAMControlAPI` with the motor/firmware law only and a shared
battery; gearbox friction stays solver-side. Its token and parameters differ
from the `NewtonBamDriveAPI` prims in these USDs, so the pinned component
remains the implementation until that drive ships and is re-admitted.

One native probe ran the retained twelve-robot reader-only recipe with the new
bundle on the same Isaac Sim 6.2 release (Newton 1.6.1rc1, Warp 1.17.0,
`SolverMuJoCo`, GPU 1, SDK recipe, graph/read reuse, `rough_walk_e` policy with
zero commands, slow limits), changing only the bundle
([plan, root receipt, bundle receipt and native summary](evidence/microduck-isaaclab-usd-20261005/)).
It completed 800 steps for all twelve robots and exited zero; the reader-only
client route (state and camera per robot) completed and closed; no teardown
error, no withheld tick. The receipt records, per robot, the fourteen parsed
`DriveBam` prims with their authored deployment attributes, the fourteen removed
drives, the seventeen coefficients checked against the m6 fit, the ground left
outside any collision group, the imported Isaac joint properties before the M6
replacements and the sole proxies in the support contract; the model identity
carries `bundle_kind: external-usd` and `asset_variant: allcollisions` and
consumes exactly `microduck_allcollisions.usd`. All twelve robots settled from
the 0.125 m spawn height to 0.1172 m and held it, with 60 steady sole-ground
contact constraints per robot and no other contact pair; the retained converted
bundle settles to 0.117 m with 84 sole-ground constraints under the same recipe.
This is simulation evidence that the asset admits and stands; it is not a
control, deadline, locomotion or physical result, and the velstand standing,
walking, braking and reset campaigns remain to be rerun against this asset.

The retained ±30 mm MCP campaign was then rerun on this asset with the same
recipe as the retained episodes (single robot, `rough_walk_e`, slow controller
limits, `microduck_distance_native_slow` profile, solver graph and solved-read
reuse) on the admitted 6.2 release
([foundation, forward and reverse audits](evidence/microduck-isaaclab-usd-20261005/locomotion/summary.json)).
The foundation episode bound identity `1d95b116…`, kept support known on every
record, produced no forbidden contact (the converted bundle's foundation had
transient non-sole rows before 0.09 s) and closed naturally. Forward +30 mm
moved +27.04 mm with −2.68 mm lateral and −0.045 rad heading during execution,
settled, did not fall and its subsequent emergency stop was confirmed; the
retained converted-bundle forward episode measured +26.21 mm and −0.043 rad.
The verdict is nevertheless **refuted**: after completion the robot kept turning,
−0.28 rad of yaw over the post-completion observation interval against the
profile's 0.08 rad heading-drift limit, the same post-completion heading drift
the status index records for the historical forward cases under the current
contract. Reverse −30 mm was **confirmed** (−25.68 mm, −0.033 rad during
execution, +0.035 rad after completion) with a confirmed stop. The asset
therefore reproduces the known forward drift problem rather than removing it;
turns, interruption, disconnect, reset and repeated starts were not rerun.

A lab-only PhysX smoke ([record and script](evidence/microduck-isaaclab-usd-20261005/physx-smoke-01.json))
loaded the same USD under Isaac Sim 6.2 PhysX with the fourteen Newton actuator
prims deactivated and the authored zero-gain drives untouched: PhysX parsed one
articulation of 14 DOFs and stepped 400 updates without error while the
unactuated robot collapsed from 0.100 m to 0.041 m. This shows the asset is
PhysX-parseable; the native CLI still refuses PhysX for BAM and nothing about
PhysX locomotion is claimed.

### Isaac Lab Newton policies as opt-in candidates (5 October 2026)

An operator-supplied archive of twelve Isaac Lab MicroDuck task exports trained
on Newton (rsl_rl 5.4.1, exported 1 September 2026, each `obs float32[1,61]` to
`actions float32[1,14]` with embedded observation normalization) was pinned by
content: the archive, each member, its external weight file and its metadata
digest, and the single-file merge that inlines the weights (zero output
difference over 256 probes; CASCADE loads a policy from one verified byte string
and cannot read external tensors). `velocity_flat` (iteration 49,999, the
checkpoint Isaac Lab #8267 calls the original Lab checkpoint) and
`velocity_rough` are registered in `assets/microduck/policy-candidates.json` as
`isaaclab_velocity_flat` and `isaaclab_velocity_rough` with the same fixed
contract as `rough_walk_e`; the other exports stay unregistered. The Isaac Lab
task pins the MJCF joint order, the 61-wide term layout
(angular velocity, projected gravity, joint position minus the stand pose, joint
velocity, previous action, twist, head pose, body pose) and
`default_joint_pos + 1.0 × action` targets, which are exactly CASCADE's
`POLICY_JOINTS`, `observation()` and `direct-v1`; the stand pose equals
`HOME_Q`. The license of the exports is not declared in the archive and none is
inferred; the training actuator model is the Lab task's at export time, before
the BAM pull requests, not the CASCADE BAM runtime.

With `isaaclab_velocity_flat` on the Isaac Lab USD asset and the same ±30 mm
recipe ([audits](evidence/microduck-isaaclab-usd-20261005/locomotion/summary.json)):
the foundation bound identity `2c72c61e…` standing at 0.1171 m with no forbidden
contact; forward +30 mm moved +25.62 mm with +0.006 rad heading during execution
and only +0.13 rad after completion, better than `rough_walk_e` but still above
the 0.08 rad limit, so refuted; reverse −30 mm moved −27.63 mm during execution
but kept walking to −48.55 mm after completion, refuted for excess progress;
neither fell and both emergency stops were confirmed. Across the four walks with
two velocity policies the picture is consistent: both track the requested
distance while commanded and both stand once the stop latches, but both keep
stepping, drifting or advancing while the controller holds a zero twist after
completion. The Isaac Lab task trains with two percent standing environments, so
a zero command is barely in distribution for these policies; the standing
checkpoint, conversely, does not track distance. The open design item is the
handoff after completion (zero-twist walking policy, a target hold, or a switch
to the standing policy), to be evaluated with the unchanged verifier; no
threshold was adjusted and no physical claim is made.

### Standing handoff at zero twist, measured (5 October 2026)

`cascade.control.microduck_handoff.StandingHandoff` is the opt-in evaluation of
the third option above. It presents the stepper's single-policy interface and,
once per attempt inside the policy slot, selects the network from the twist that
attempt observes: the admitted motion policy whenever the commanded twist is
nonzero, the official VelStand whenever it is exactly zero (no lease, a completed
distance, a latched stop). The previous action carried into an observation is
the last committed raw action whichever network produced it; the bounded retry
after a crossed stop re-selects with the latest intent; cadence, the targets
transform, limits, stop semantics, support checks and the verifier are
untouched. The bridge admits it only when `--handoff-profile
stand-on-zero-twist-v1`, `--standing-policy` and `--standing-policy-sha256` are
given together; the standing policy must match the pinned VelStand and differ
from the motion policy; the model identity gains `recipe.handoff` (profile,
rule, both digests) only in that case, and receipts record the switches and the
commits per network.

It was measured natively on the Isaac Lab USD asset with the same ±30 mm recipe
(GPU 1, Newton 1.6.1rc1; [audits](evidence/microduck-isaaclab-usd-20261005/locomotion/summary.json),
[braking analysis](evidence/microduck-isaaclab-usd-20261005/locomotion/post-completion-analysis.json)).
`isaaclab_velocity_rough`, the export its authors recommend, was also run
without the handoff:

| Walk | Executed mm | After completion mm | Yaw after completion rad | Verdict |
|---|---:|---:|---:|---|
| `rough-forward01` (`isaaclab_velocity_rough`) | +25.86 | +26.94 | +0.006 | confirmed |
| `rough-reverse01` (`isaaclab_velocity_rough`) | −26.65 | −55.39 | +0.032 | refuted (excess progress) |
| `roughho-forward01` (rough, handoff) | +25.68 | +28.14 | +0.053 | confirmed |
| `roughho-reverse01` (rough, handoff) | −25.64 | −46.81 | +0.008 net, 0.080 peak | refuted (wz drift; the first violation is kept) |
| `flatho-forward01` (flat, handoff) | +26.50 | +27.57 | +0.127 | refuted (wz drift) |
| `flatho-reverse01` (flat, handoff) | −27.33 | −41.76 | +0.107 | refuted (wz drift) |

Every switch was clean: two per walk, every attempt committed, and the two
handoff foundations, which evaluate VelStand alone, reproduce each other's
trajectory exactly. The verdicts nevertheless match those without the handoff.
The per-step physics logs explain why: the recipe commands 0.3 m/s for 30 mm, so
each walk completes after 14 to 17 policy slots (0.28 to 0.34 s), mid-stride,
with the body moving at 0.09 to 0.23 m/s. Whatever network then holds a zero
twist needs 0.7 to 0.9 s to come to rest in reverse and is 15 to 26 mm further
along the command two seconds later; the handoff shortens that reverse overrun
by 4 to 7 mm (rough 25.6 to 21.3 mm, flat 22.1 to 15.2 mm) and leaves the flat
forward yaw unchanged (0.123 rad with and without it). The standing-only
foundations do not drift (yaw within 0.02 rad over four seconds, none of it in
the second half), so the residue belongs to the stride in flight and the braking
that follows, not to standing. Braking is a policy behaviour, not a fixed limit:
`rough_walk_e` came to rest in 0.28 s and 5 mm from the same 0.22 m/s in
`usd-reverse01`. The handoff therefore remains an opt-in software candidate, not
a remedy. The next candidates are a checkpoint that sees zero and decelerating
commands in training, or a distance-aware speed in the walk skill, both to be
judged by the unchanged verifier. Nothing physical is claimed.

### The braking residue reproduced in Isaac Lab itself (5 October 2026)

To separate the policies from CASCADE's pipeline, the same ±30 mm protocol was
run directly in Isaac Lab's own MicroDuck environments, outside CASCADE
([report](evidence/microduck-isaaclab-repro-20261005/REPORT.md),
[aggregates and per-episode table](evidence/microduck-isaaclab-repro-20261005/report.json),
[per-step trajectories](evidence/microduck-isaaclab-repro-20261005/trajectories.csv.gz)):
AntoineRichard/IsaacLab `antoiner/feat/microduck-rough-velocity` at
`eafc80df` (the #8161/#8265/#8266/#8267/#8270 stack over upstream `develop`
`4aa39c10`), installed kit-less with the fork's own `uv sync` (Newton 1.6.1rc1
from the `release-1.6` branch, MJWarp 3.12.0, warp 1.17.0), tasks
`IsaacContrib-Velocity-Flat-MicroDuck` and `IsaacContrib-Velocity-Rough-MicroDuck`
built exactly as `isaaclab play` builds them (play mode: noise and pushes off),
one environment on the local mirror of the Nucleus `microduck_walk.usd`, the
native BAM servos, and `velocity_flat.onnx` / `velocity_rough.onnx` (iteration
49,999, the same digests CASCADE admits) run through onnxruntime on the
environment's own 61-value policy observation. Protocol per episode: 1 s settle
at zero twist, (±0.3, 0, 0) until the root has moved 25 mm along its initial
heading (closed loop; open-loop 15-step and 1 s variants too), then (0, 0, 0)
for 3 s. Three configurations: Lab's play-mode randomization and BAM deployment
(6.5 to 8.2 V, sag, 3 to 6 step command delay, 1.75 A limit), everything
deterministic, and CASCADE's nominal BAM profile (7.4 V, sag 0.1, no delay, no
current limit). 36 walk episodes, 8 steady-gait braking episodes, 9 ten-second
standing episodes, no fall.

The overrun is there in every configuration: after the zero command the robot
keeps moving in the commanded direction, forward +4 to +31 mm (mean +20 mm flat,
+15 mm rough), reverse −11 to −46 mm (mean −25 mm flat, −29 mm rough), with a
5 to 27 mm lateral swerve, so a "30 mm" walk ends 30 to 59 mm forward and 36 to
71 mm in reverse; the reverse-heavier asymmetry CASCADE measured (flat
−27.6 → −48.5 mm, rough −26.7 → −55.4 mm) sits inside that envelope. The
mechanism is the one the braking analysis above describes: the threshold is
crossed while the body still moves at 0.14 to 0.26 m/s forward and 0.27 to
0.34 m/s in reverse (the policy first swerves sideways, then crosses the
threshold at its first-stride velocity peak), and the network needs 0.3 to 0.5 s
to brake; from a steady 0.19 to 0.21 m/s gait it stops in 0.42 to 0.68 s and
+2 to +17 mm forward, −11 to −21 mm reverse. It never drifts indefinitely: every
episode is still (planar speed below 0.02 m/s) 0.42 to 0.74 s after the zero
command (one outlier, 1.34 s), moves under 0.4 mm in the last second, and a
zero command from rest holds 10 s with under 0.35 mm drift and 0.25° yaw. The
61-value observation CASCADE assembles matches Lab's term by term (order, units,
`HOME_Q` equal to the Lab default pose, raw previous action; verified
numerically on the Lab logs); the one structural difference is Lab's constant
one-step joint-velocity delay, which is part of the training recipe. The
deployment parameters move the forward overrun (CASCADE's nominal BAM profile
gives +4 to +11 mm, closest to CASCADE's own +1 mm) but not the reverse one.
Conclusion: the post-completion residue is a property of these checkpoints and
their training command distribution, not of CASCADE's observation, actuator or
stop pipeline; the remedies stay the two named above. Not covered: Kit itself
was not started (kit-less Lab on the same Newton release Isaac Sim 6.2 bundles),
the rough task ran on the collision plane because MJWarp rejects an all-flat
mesh terrain, and nothing physical is claimed. The driver, generalized, was
offered to the fork as [AntoineRichard/IsaacLab#21](https://github.com/AntoineRichard/IsaacLab/pull/21).

## Remaining admission work

The current real MCP campaign confirms supported standing and a separately
observed stop. Forward/reverse tracking is refuted and both turns expire.
Interruption and disconnect invalidate admitted commands; braking from an
established gait remains unproved. Small forward and reverse commands also
failed tracking in an earlier upstream reference. The
[startup research](research/microduck-policy-startup-2026-10-02.md) reports a
related upstream issue and its limits; the
[BAM follow-up](research/microduck-bam-2026-10-02.md) distinguishes the pinned
actuator from incompatible newer fits. Neither establishes a remedy.
The [full-mjlab diagnostic](research/microduck-fullenv-startup-2026-10-02.md)
also records negligible +0.1 m/s motion from rest and after a +0.3 phase,
with the same official policy and unchanged production limits.

Required acceptance includes current-source equilibrium, forward/reverse
tracking, turns, interruption, disconnect, reset, repeated starts, and video
from the same episode. Each advertised engine needs its own evidence.
PhysX currently lacks the separate previous-step external generalized load
required by the chosen BAM contract; tested friction and contact primitives
alone do not provide that contract. The native CLI rejects PhysX explicitly.

Models remain external: their notices declare BY-SA-NC without a version;
code, weights and the BAM implementation retain their separate notices.
This change does not vendor meshes, USD or ONNX files. The implemented
[conversation gateway](CONVERSATION.md) retains the ownership boundary from the
[earlier hosted conversation design](MICRODUCK_CONVERSATION_DESIGN.md).

### Shared-scene implementation boundary

`sim/microduck_shared.py` binds explicit robot namespaces to disjoint native
coordinates, body/shape identities and a single world. Its coordinator previews
all independent policies without changing their history or native targets, then
checks the common completed clock and each permission under a shared memory
fence. Only an admitted group commits its histories, applies the owned BAM groups
and requests **one** solve. A changed command before that fence discards the whole
preview group and retries once against the same completed state. Repeated
revocation withholds the tick without consuming a step or publishing a frame;
wall, signal and freshness guards still apply on each attempt. Attempt/evaluation
records include discarded work, while solve, history and BAM counts do not.
A stop after admission applies to the next regular control slot: the in-flight
solve and native delay cannot be rolled back. A partial commit or native fault
still contains the scene; stop acknowledgements do not prove physical rest. All solved contact
rows remain visible to each robot's existing support checker, so another robot
cannot count as ground. Model identities and command epochs remain per robot.

The shared owner captures and validates global state/contact channels once for
each completed scene, then detaches each robot's joint and body observations.
The capture binds the model, current state, contact buffer, solver data and
completed clock before reading; a changed binding rejects the entire capture.
No snapshot is reused across solves, and each robot still receives the complete
contact evidence. CPU tests check channel read counts and exact data equivalence;
they do not establish a native speedup.

Within that completed scene, support is decoded once into immutable contacts
and rebound to each robot's model and epoch without reconstructing every
contact. Only transitively immutable records can be shared; mutable legacy
values retain copy isolation. Public observations and physics logs preserve the
complete dictionary schema, while each reply remains detached. Layout, shape,
clock and source-admission checks still gate reuse; permission and age remain
live on every controller read. This reduces repeated CPU work and does not
establish a control deadline or physical-stop result for twelve robots.

The [retained support diagnostic](evidence/microduck-shared-20261004/support-profile.json)
completed 800 solves, 2,400 policy commits, nine overview/support-probe pairs and
24 reader calls with twelve robots; all owned processes and clients closed
naturally. Owner-attempt durations had a median of 114.911 ms, p95 of 130.478 ms
and maximum of 655.533 ms. **22 attempts exceeded 400 ms**, so the original
control deadline remains unresolved. This run sent no stop, reset or motion
commands and does not supersede the retained twelve-robot control failure.
These times describe this episode; they do not establish a causal speedup.
The earlier prelaunch refusal remains recorded alongside the completed run.

The subsequent [pinned zero-command trial](evidence/microduck-shared-20261004/support-control.json)
also completed 800 native solves, but **its twelve-robot control sequence failed**.
The first robot completed its zero-command execution; the second timed out while
requesting a fresh preflight state, before sending its command. Its stop was
acknowledged, the client closed, and the remaining ten controls were not attempted.
All physical outcomes remain unverified. The original 0.5-second RPC and
0.4-second no-progress limits were retained; a 691 ms publication gap crossed
the failure window. The timing evidence does not establish the cause.

The shared stepper also retains immutable copies of the full legacy contact
pairs and constraint addresses for that solve. Public native, robot-view and
controller reads still return fresh lists; the private recording path keeps
the same JSON arrays, order and bytes. Nonplain legacy values retain ordinary
deep-copy isolation. This removes repeated list construction within a solve,
without changing GC settings, validity checks or the public state contract.

CPU tests cover 1, 2 and 12 synthetic participants, including index permutations,
command isolation, cancellation and contact vetoes. The ordinary one-robot
entrypoint retains its control cadence.

`scripts/isaac_microduck_shared.py` now supplies a bounded native foundation:
one shared scene, separate ONNX/BAM histories, disjoint robot views, a common
completed-step overview and per-robot content identities. It takes the native
bridge's explicit input flags plus `--robots 1..12 --spacing 2`. Use `--port 0`,
`--solver-cuda-graph --reuse-solved-read` and the explicit SDK recipe; this
foundation opens no command sockets. Every run needs a fresh output directory
and an outer process deadline. Shared solver storage scales with robot count;
contact observations retain every solved row.

The [retained native evidence](evidence/microduck-shared-20261004/foundation.json)
records separate source-bound runs with **1, 2 and 12 robots**. Each completed
800 global physics steps (4 simulation seconds), 200 policy evaluations per
robot, and 9 overview/force-probe pairs. Commands stayed zero, no robot fell,
clocks and identities matched, and owned processes closed with native exit 0.
The twelve-robot run retained measured sole support for every robot, with
maximum XY drift below 2.8 mm and maximum tilt below 0.014 rad. These are
initialization/balance results, not walking, fleet task or real-time acceptance.
The earlier twelve-robot attempt failed closed on insufficient contact storage
and is retained alongside its corrected retry.

An opt-in candidate, `--serve-base-port 0`, exposes separate loopback endpoints
after the first completed state and matching overview; a positive value chooses
the first port in a consecutive range. `BRIDGE_LISTENING.json` maps robot IDs to
ports and bound identities. Each endpoint retains the existing reader, control,
renew and stop channels. Commands and resets wait in a bounded per-robot queue
until the simulation owner drains it before a tick, preserving their original
receipt deadline. Expired, cancelled or closed requests cannot run on a later
tick. A nonblocking control-channel guard rejects observed EOF or invalid
pipelining before admission, after the mutating callback and before the reply;
discarding an admitted reset restores its latch. Stop bypasses the queue and
withdraws pending commands immediately; a stop crossing pure preview invalidates
that group without faulting healthy peers. Stop ACKs do not
prove physical rest. CPU/TCP regressions cover these contracts, partial listener
startup rollback and shutdown with pending commands.

The [endpoint evidence](evidence/microduck-shared-20261004/endpoints.json) records
fresh identity probes with **1, 2 and 12 robots**, each completing 800 shared
solves, 200 policy commits per robot and 9 overview/force-probe pairs. Separate
1- and 2-robot episodes completed state/camera reads, stop, reset and a zero
velocity command through `FleetRuntime -> RobotRuntime -> locomotion -> SafeBase
-> IsaacBase`. The two-robot run recorded two naturally discarded previews at
one permission change; the retry committed against the same completed state.
Adversarial timing barriers were exercised only in CPU tests.

The twelve-robot control episode also completed all 800 native solves with known
support and no falls, but **its control sequence failed** on the first robot:
the admitted zero command received feedback through step 11, then exceeded the
unchanged 0.4 s progress deadline. The measured publication gap to step 12 was
0.705 s; the stop latch was acknowledged and the remaining eleven commands were
not attempted. Native completion does not make this a control pass. The evidence
retains this failure, the earlier preparation-race failure, the earlier
794-step wall timeout and all prelaunch refusals. Completed native scopes closed
naturally; input files and protected stores remained intact. No performance
comparison is claimed across changing GPU occupancy.

For a separately admitted diagnostic, `--profile-phases` records inclusive
owner-thread spans in `timing.jsonl`: completed reads, support/native capture,
policy preparation, BAM, solve, publication, recording and camera/probe work.
Rows retain the attempt's completed clocks and solved/withheld/error outcome,
plus the preceding profile write/flush cost; the footer retains the final cost.
Nested durations overlap and must not be summed. Reader calls are not wrapped.
The flag installs no wrappers when disabled and is bound in the effective model,
so enabling it requires a fresh identity probe and pin. Profiling keeps the
existing control gates, cadence, contact evidence and physical records.

The [retained twelve-robot phase diagnostic](
evidence/microduck-shared-20261004/phase-profile.json) completed 800 solves,
2,400 policy commits, 24 reader events and 9 overview/support-probe pairs, then
closed naturally with exit 0. Median attempt duration was 343 ms (p95 831 ms).
Inclusive per-attempt medians were 82 ms for preparation, 101 ms for completed
state validation, 79 ms for publication and 55 ms for constructing/writing the
physics record. These overlapping categories are not additive. Large pauses
occurred in several phases; their cause remains unresolved. The first 160
attempts overlapped the recorded scope of a separate RTX probe on GPU0. This
diagnostic neither admits twelve-robot control nor establishes a native speedup.

Physics rows share only private contact projections during one JSON encoding;
public controller responses remain detached. This moves projection work into
`write.physics.jsonl`, within the inclusive `record.physics` and attempt spans.
Compare those enclosing spans for total work: the write span includes encoding,
so its change alone does not measure file I/O or GC cost.

The phase profile also retains passive Python GC callbacks from all threads,
with absolute timestamps, event IDs, generation and collection counts. Each
attempt and the footer drain a persistent queue bounded to 4096 events, including
events outside owner attempts; starts and stops may cross rows. Cumulative ticket
counts expose overflow or callbacks still in flight, without waiting for them.
Missing/error events or a removed callback prevent complete correlation claims.
Draining callback metadata is outside the attempt and write/flush durations.
GC settings are only observed, and closing the profile removes only its own
callback. Callback intervals include scheduling
and other callbacks; they do not measure isolated CPU cost or establish a cause
for a control timeout. Profiling does not disable or force collections.

The trigger diagnostic also copies at most 16 Python frame locations on each
generation-2 start, with paths and function names limited to 512 characters.
It retains no frames, locals or source lines; absent Python callers and truncated
stacks are explicit. This identifies the observed trigger, not the heap scanned.
Collector counts/statistics and allocated-block counters bracket callback
registration; they are not a heap-object census or SDK-startup measurement.
Stack and counter capture durations are recorded, but exclude callback dispatch,
queue and encoding overhead. They do not measure the observer's entire cost or
justify subtracting that cost from GC intervals. The recipe uses a distinct
profile identity, and counter/capture errors prevent complete attribution.

The [retained trigger diagnostic](evidence/microduck-shared-20261004/gc-trigger.json)
completed 800 shared solves, 2,400 policy commits, 24 reader events and 9
overview/support-probe pairs, with normal closure and complete callback
accounting. All ten generation-2 starts had stacks: eight in deep-copy paths,
one in plain-record tuple construction and one in support validation. Eight
stacks reached the 16-frame limit. These are observed trigger sites, not evidence
of which objects were scanned or what caused the pause.

Stack extraction took 0.020 ms median and 0.024 ms maximum, excluding the other
observer costs described above. Generation-2 intervals remained about 475 ms
median; ten attempts exceeded 400 ms, with a maximum attempt of 615 ms. This
reader-only diagnostic did not test control recovery, change collector settings
or alter the three retained twelve-robot control failures.

The [legacy-contact sharing reader](evidence/microduck-shared-20261004/legacy-contacts.json)
completed the same 800 solves, 2,400 commits, 24 reader events and 9 capture/probe
pairs with normal closure. Attempt durations were 83.40 ms median, 91.54 ms p95
and 595.44 ms maximum. One attempt exceeded 400 ms, overlapping a 508.06 ms
generation-2 interval; the preceding trigger reader retained ten such attempts.
Its thirteen-frame stack points to `dataclasses.fields` during private physics
JSON projection. This observed trigger does not identify the heap scanned.
The integrated source includes the optional RTX-cache feature at its inactive
default. Both GPU admission cohorts were empty, but host load was uncontrolled;
these separate episodes establish neither causal speedup nor a control deadline
guarantee. The original control limits and earlier failures remain intact.

The [subsequent fixed twelve-robot zero episode](
evidence/microduck-shared-20261004/legacy-control.json) completed all 72 client
events: stop, reset and one zero command per robot, with `execution_ok=true`
and physical outcomes `unverified`. It completed 800 solves, 2,400 policy
commits and 9 capture/probe pairs, then closed normally. Two naturally
invalidated preview cohorts discarded 24 evaluations and retried against the
same completed state; there were no withheld solves or extra history commits.

Its only generation-2 interval started 31.84 s after the final command result
and 31.72 s after the final cleanup stop ACK. The longest attempt was still
575.06 ms, so this protocol pass did not exercise control through that pause or
establish worst-case deadlines, physical rest or gait. The episode kept the
original 0.5 s RPC, 0.4 s progress and 0.5 s state-age limits, with no favorable
retry. The three earlier twelve-robot control failures remain retained.

Two [reader-only GC profiles](evidence/microduck-shared-20261004/gc-serialization.json)
each completed 800 solves, 2,400 policy commits, 24 reader events and 9
overview/support-probe pairs, with normal native and client closure. The second
used the plain-record serializer; its effective recipe changed only that source
file. The recorded attempt durations were:

| Profile | Median | p95 | Maximum | Attempts over 400 ms |
| --- | ---: | ---: | ---: | ---: |
| GC baseline | 117.15 ms | 161.18 ms | 646.41 ms | 22 |
| GC with plain-record serialization | 110.75 ms | 127.60 ms | 655.07 ms | 22 |

Both traces had complete callback accounting. Generation-2 GC intervals
overlapped all 22 slow attempts in each run; this establishes temporal overlap,
not an isolated CPU cost or a causal speedup. The comparison started after the
separate voice scopes closed. Both launch admissions recorded the same foreign
GPU process identity.
The original 0.5 s RPC and 0.4 s progress limits remain unchanged and unresolved
for twelve-robot control. These profiles requested only observations; the
earlier control failures remain retained.

The subsequent [private-row encoding profile](
evidence/microduck-shared-20261004/row-serialization.json) also completed
800 solves, 2,400 policy commits, 24 reader events and 9 overview/support-probe
pairs, with normal closure. Attempt durations were 105.59 ms median, 116.03 ms
p95 and 662.65 ms maximum. Ten attempts exceeded 400 ms; each overlapped a
generation-2 GC interval, with complete callback accounting. The preceding
serializer profile retained 22 such attempts. GPU admission changed from a
foreign process to an empty cohort, so this comparison does not establish a
causal speedup. The projection work moved inside the write span as described
above. The remaining long attempts still exceed the original control limits;
this reader-only episode grants no control or physical-stop acceptance.

The [subsequent bounded control episode](
evidence/microduck-shared-20261004/row-control.json) **failed** during `duck02`'s
reset of the stop latch: its reply exceeded the unchanged 0.5 s RPC limit.
`duck00` and `duck01` completed zero commands with unverified physical outcomes;
no walk was dispatched for `duck02`, and the remaining nine control sequences
were not attempted. The failure window overlapped a 605 ms attempt and a
generation-2 GC interval; the measured publication gap was 601 ms. Timing overlap
does not establish isolated GC cost or a cause.

All 800 native solves, 2,400 policy commits and 9 overview/support-probe pairs
completed, and client resources and native processes closed normally. The
`duck02` cleanup stop lacked an ACK because its owner channel was disconnected,
with delivery marked uncertain. Completed states show its latch set from step
21 onward; they do not reconstruct whether reset ran between states or establish
physical rest. This remains a control failure despite native completion, and
both earlier twelve-robot control failures remain retained.

**Opt-in startup-heap freeze (software candidate, 5 October 2026).** Every
long attempt above overlapped a generation-2 collection of roughly 475 ms. A
full CPython collection traverses every tracked container in the oldest
generation, which an Isaac Kit process fills during SDK, scene and identity
startup; reducing per-attempt garbage changes how often such collections start,
while only a smaller retained tracked population shortens them, and the startup
heap is the dominant retained population. `scripts/isaac_microduck_shared.py --gc-policy
freeze-startup-heap` records the selected policy in the hashed configuration
before identity binding, then, after stepper construction and before
`fleet.start()`, runs one explicit full collection and `gc.freeze()`
(`src/cascade/sim/heap_freeze.py`), so later automatic collections traverse only
objects allocated after that point. The collector stays enabled with unchanged
thresholds and callbacks, and the passive phase profile observes the explicit
collection as an ordinary event. `runtime.json` (`backend.gc_policy`) and
`receipt.json` (`gc_policy.applied` / `gc_policy.released`) retain counters,
durations, the policy implementation digest, the interpreter's own frozen
immortal objects and the policy's delta; the selection is part of the hashed
configuration and `src/cascade/sim/heap_freeze.py` joins the bound source
hashes. Release follows endpoint and fleet closure and precedes SDK shutdown,
also after a teardown fault or an apply interrupted after `gc.freeze()`. The
[CPU mechanism receipt](../benchmark/results/heap_freeze_cpu_20261005.json) on
a synthetic 2.3-million-object heap measured a 200.3 ms median full collection
before the freeze, 0.001 ms with nothing retained afterwards, 10.6 ms with
100,000 retained rows and 67.9 ms with 400,000. That is a collector measurement on
one host, not a control result: the reader-only probes below ran with the policy
but exercised no command, the 0.5 s RPC and 0.4 s progress limits are unchanged,
and one Factory readiness failure occurred without any generation-2 interval, so
the policy cannot explain every retained failure. Native control evidence must
show sustained command activity overlapping observed collections under the
original limits.

**Direct reader-only probes with and without the policy (5 October 2026).**
Four direct launches of the retained twelve-robot reader recipe on the working
tree ([evidence and hashes](evidence/heap-freeze-20261005/summary.json); same
bundle, policy, limits and SDK recipe `isaac62_48b2d951`, Kit Python 3.12.14,
GPU 1, seeded private caches, private learned stores, systemd user scope) all
completed 800 solves, 24 reader events and ordinary closure. The baseline run had
one 515.7 ms generation-2 collection on the owner thread during attempt 199, and
that attempt, a camera overview, was the only one above 400 ms at 1344.8 ms
(median 80.9 ms, p95 89.3 ms). In the three runs with `--gc-policy
freeze-startup-heap` the explicit pre-freeze collection took 464.8, 468.4 and
479.2 ms and froze 1,254,076 objects in 0.023–0.025 ms before `fleet.start()`; no
generation-2 collection occurred during any of their 800 attempts, whose maxima
were 208.4, 212.1 and 202.6 ms (medians 84.8, 79.7 and 83.3 ms). At release,
after the owner loop had closed, a full collection over everything the episode
retained took 2.329 and 2.134 ms while frozen in the two runs that measured it
(6,786 tracked objects outside the frozen set, 74 unreachable), while the
collection after unfreezing took 461.0–488.2 ms across the three runs; 673
frozen objects had already been freed by reference counting in each. The first two policy runs used earlier revisions of the helper and the
third the final implementation; the summary records each implementation digest.
These are single reader-only runs on an uncommitted tree with recorded file
hashes, not reviewed plans: no command, stop or deadline recovery was exercised,
so the retained 0.5 s RPC control failure is untested under the policy; the
attempt medians establish no speedup; and the 2.1–2.3 ms figures are collections
at close, not inside an attempt.

These endpoint outcomes remain physically unverified. Walking, fleet tasks,
shared-space interactions and measured individual/global motion stops still need
their own acceptance evidence. The [original twelve-robot overview](
evidence/microduck-shared-20261004/endpoints-twelve-zero.jpg) comes from the
read-only probe at completed step 802; every velocity command stayed zero.

The overview uses a widely separated grid; its small floor display
rectangle does not describe the extent of the physical infinite plane.

### Shared-owner per-step cost: one host snapshot for the BAM checks (7 October 2026)

Every `.numpy()` of a device array is a full device sync. In the shared owner,
each of the twelve `NewtonBamAdapter`s re-read the same ~25 whole-model Warp
arrays (model properties, solver DOF map, state, `nefc`) on every step to run
its fail-closed checks: about 400 syncs per 5 ms step. `BamHostSnapshot`
(`control/newton_bam.py`) is one per-step host copy of those arrays, captured by
`SharedMicroduckStepper.tick` after the policy commits and before the first
actuation and handed to every adapter's `before_step(dt, snapshot=…)`. Every
adapter still runs every check, against the same bytes a private read would
return; a snapshot of another stage or step is refused, and model drift, NaN
state and DOF-map changes still fail closed through it. The single-robot path
is unchanged (it captures its own snapshot). CPU tests assert identical efforts
with and without the snapshot and one device read per world array per step.

The [same-day A/B](evidence/microduck-owner-host-snapshot-20261007/REPORT.md)
(twelve robots, one world, route profile, twelve closed-loop 5 m clients, x86
rig) measured `bam.before_step` 24.3 → 13.6 ms per step (1.87 → 0.99 ms per
robot) and the step median 76.2 → 61.3 ms, other phases unchanged. The new
`--profile-sync-solve` diagnostic (with `--profile-phases`) attributes the GPU
time of the physics step to the `solve` span instead of the first host read:
**7.6 ms of GPU physics per 5 ms step** for twelve robots, previously hidden in
`support.decode` (whose pure decode cost is 4.4 ms). All twelve client walks (one per robot) remain
`unverified` on the unchanged deadlines; this is a measured cost reduction, not
a control or admission result. The owner cannot reach real time while the
physics alone takes 1.5× the step; the remaining host cost is in
`completed.validate` (decode, native capture, per-robot validation and deep
copies), the per-adapter output reads and Warp launches of `bam.before_step`,
`policy.prepare` and `publication`.

### Shared-owner per-step cost: cheaper per-robot binds and reader polls (7 October 2026)

An in-process `cProfile` of the owner (`--profile-cprofile`, with
`--profile-phases`; the stats are dumped before the SDK shutdown, which does not
return) over 1000 completed steps showed pure-Python work repeated per robot or
per reader poll on data that is immutable within a step:
`RobotBinding.model_identity_sha256` recomputed `asdict + json + sha256` on every
access; `SceneLayout.robot_support` rebuilt the shared `SupportObservation` with
`dataclasses.replace` for each robot, re-running `__post_init__` over every solved
contact; every client `state()` poll (about one per robot per step, served on the
owner's GIL) walked the ~90 contacts again in `BaseState.as_dict`; and
`identifier()` scanned every shape path and joint name character by character.
Now the digest is a `cached_property` of the frozen binding (`replace` yields a
new binding and a new digest); `SupportObservation.rebound` validates only the two
identifiers it changes and keeps the shared, already validated contact tuple
(non-plain records keep the `replace` path); `BaseState.as_dict` memoizes the plain
copy of the contacts once per shared contact tuple and still hands every reader
fresh containers; `identifier()` applies the same rule through a compiled regex.
No check was removed and every reader reply is content-identical and isolated;
tests count the walks and the re-validations.

The [same-day A/B](evidence/microduck-owner-reader-rebind-20261007/REPORT.md)
against the host-snapshot build (same recipe as above) measured the step median
61.3 → 48.55 ms and p95 147.6 → 126.3: `completed.validate` 29.7 → 23.8 ms
(0.57 → 0.28 per robot), `policy.prepare` 8.5 → 4.2, `publication` 5.4 → 3.0,
`bam.before_step` unchanged at 13.9. All twelve client walks remain `unverified`
on the unchanged deadlines. What remains: the GPU physics step (7.6 ms per 5 ms
step), `bam.before_step` with ~230 Warp device→host copies per step (the twelve
adapters' five output reads each are 60 device syncs; a cohort-level batched
finiteness check is the next slice), the deep copies of `completed.validate` and
`policy.prepare`, and the sparse `record.physics` / `camera.overview` writes that
make the p95 tail.
