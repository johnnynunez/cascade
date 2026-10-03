# Mounted Factory fastening runtime

Implementation checkpoints, 2026-10-03. The optional mounted domain now has
an ordinary configuration/MCP route and an inert shipped profile with unresolved
device and model pin. CPU contracts pass; no native result admits this new runtime yet.
The [ordinary wrist routine](FASTENING_RUNTIME_GAP.md) remains unverified.

`FasteningDomain` in `skills/fastening_runtime.py` can be passed directly to the
ordinary `RobotRuntime({"fastening": domain})`. It declares one command resource
covering the complete arm and spindle; a second writer of the same endpoint is
rejected by the normal resource catalog. Its only motion operation is
`fastening.turn_screw(turns=1, direction="tighten")`. The configured socket is
already mounted and the nut already engaged. It has no pickup, acquisition,
loosening, seating or preload capability.

`control/fastening.py` supplies the separate controller boundary. Its immutable
binding includes the recipe digest, exact body/tool/joint/collider identities,
permitted contact pairs, solver timestep and the digest of numerical limits.
This version explicitly implements the fixed world +Z Factory M20 fixture with
2.5 mm pitch; it is not an arbitrary-thread adapter. The native owner must check
the binding against its actual constructed model. A dictionary of labels or a
claimed model hash alone does not establish native fidelity.

Every `FasteningSolve` retains its original local monotonic capture time, epoch,
generation, step, physical time, joint positions/velocities/actuator efforts,
conservative arm/tool collision bounds, actual nut/fixture/tool poses, body and
spindle speeds/effort, commanded spindle effort and all solved collider pairs.
Full solver count must match the records. Unknown buffers and counts at capacity
are refused. Thread/tool witnesses are recomputed against the bound registry;
zero-force records cannot create engagement. Any positively loaded forbidden
pair rejects. Native collision/constraint overflow checks remain the producer's
obligation; these Python structures cannot certify the SDK's channel coverage.

The passive journal preserves every solve for each reader cursor. Reading does
not solve physics or refresh timestamps. Missing steps, epoch/clock changes,
buffer overrun and producer failures are visible; recovery cannot erase a fault.
The current API is in-process and uses one host monotonic clock. A future remote
transport needs explicit clock mapping and uncertainty, not copied timestamps.

The write guard uses imported joint/effort limits and explicit task bounds:

| Check | Initial candidate contract |
| --- | --- |
| Joint speed / margin | 0.8 rad/s / 0.02 rad, from the SO-101 safety profile |
| Joint following error | 0.1 rad maximum; not a hardware calibration |
| Spindle effort / measured angular speed | 0.05 Nm / 10 rad/s |
| Geometry | Complete moving arm/tool bounds within declared workspace and above table; native target FK required |
| One-turn command budget | 4 physical seconds and 40 host seconds, both required |
| Maximum observation age | 0.2 host seconds |
| Observed rest | 0.5 continuous physical seconds, zero commanded/solved spindle effort, speeds ≤0.001 m/s and ≤0.02 rad/s |
| Rest budget | 3 physical seconds and 30 host seconds, beginning at the stop ACK |

The physical and host budgets are distinct candidate limits. They do not claim
real-time operation. No native experiment has yet admitted these new limits.
Threading still uses `ThreadContract` unchanged: 0.05-turn tolerance, 0.3 mm pitch
residual, 1 mm radial / 0.05 rad tilt gates, fixture-motion bounds and at least
80% contact-supported angular travel.

Permission is checked under the same lock as each bounded backend upload, with
epoch, generation, original wall/physical deadlines, every solve and target-rate
limits. Stop revokes authority; its immediate ACK is a latch receipt, not proof
of a zero-effort upload or physical rest. The native owner must submit exact zero
spindle effort on stopped/rejected/exception paths while retaining guarded arm
support. It must not leave the last spindle effort latched, disable the loaded
arm's torque, or claim closure when that upload/owner closure fails. No solver or
logger runs under the upload lock. An unbounded SDK upload is not made bounded
by this Python lock and needs the owner's external lifecycle watchdog.

The domain starts verification strictly after command admission. It consumes a
separate passive reader and reuses `verify_threading` on actual fastener poses,
not wrist travel or actor completion. It then stops and observes the new stop
generation through a full rest window. Replayed/negative/future stop timestamps,
wrong generations and active velocity braking cannot confirm rest. An uncertain
admission exception still invokes stop. Trace and task completion use the normal
RobotRuntime path; synthetic fixtures remain excluded from physical task credit.

Validation: 115 CPU tests passed in 2.19 seconds with CUDA disabled and one BLAS
thread. This includes 56 new synthetic cases, existing threading tests and
ordinary RobotRuntime regressions. The first 41-case and intermediate 108-case
runs also passed. Independent review added late-read, pre-ACK capture and replay
controls: a pending response can veto after a deadline but cannot earn positive
credit. One initial pre-stop test reversed the capture clock and correctly hit
that veto; its fixture now models an actual 10 ms ACK latency to isolate the
pre-fence exclusion. The original failed test receipt is retained. All 658
source/config/test hashes and protected memory remained unchanged in each run.
See [stage-one receipt](../benchmark/results/factory_runtime_stage1_20261003.json).

Next stage is concrete instrumentation/control integration with the existing
`SeatingScene`: actual native tensors and full contact buffers per subsolve,
identity-bound FK bounds for each target, one guarded uploader and independent
reader, exact zero-spindle stop plus owned closure. A new coordinated native
episode must establish parity, one measured turn and rest, with retained
negative controls and video from that episode. The older separate Factory
seating experiment is not substituted for this missing runtime admission.

### Native readback implementation checkpoint

`sim/factory_observation.py` now implements the first native adapter boundary.
It imports no optional SDK at module load. Its entrypoints inspect the actual
Newton 1.6 / MJWarp 3.12 structures and reject mismatched pinned implementation
hashes, missing arrays and ambiguous mappings. This checkpoint is callable
instrumentation, **not an enabled runtime profile or native admission**.

`collision_coverage` checks the broad phase, GJK and split work queues,
mesh/triangle/plane buffers, final contact buffer, reducer occupancy and insertion
failures. Newton's `verify_buffers=True` only prints diagnostic warnings; these
host checks raise instead. Counts at capacity are refused conservatively. The
hydroelastic route is unsupported. A separate output `Contacts` buffer requests
the real `force` attribute; `SolverMuJoCo.update_contacts` converts the completed
MJWarp solve without another solve or `forward`. Its compatibility with this
Factory scene's `use_mujoco_contacts=False` path still needs native parity proof.

The shared solved-contact decoder validates every active contact constraint
row, exact geom/shape pairs, normals, positions, force signs and constraint
coverage. Active zero-force contacts remain recorded. An inactive candidate is
explicitly marked as lacking a solved constraint, with zero normal load and no
invented point/vector. Every candidate remains in the count/identity ledger.
The independently reviewed older MicroDuck decoder is reused for these common
SDK semantics; none of its foot-specific registry or standing evidence applies
to fastening.

Joint/control/DOF indices are mapped explicitly; native actuator order is not
assumed to equal XML control order. The descriptor requires one unit-gear hinge
actuator per controlled DOF, direct-control routing, stateless bounded arm
position servos, and one direct spindle motor with the authored 0.05 Nm cap.
Gain/bias arrays, ranges, reference offsets and local joint axes are retained.
Readback uses `qfrc_actuator` and also retains actuator scalar force and native
control input. Direct `qfrc_applied` and external `xfrc_applied` must both be zero;
their effects cannot be hidden inside a motor-off claim. Constraint and passive
forces are separate raw channels. Solved forces/contact geometry describe the
interval **[previous step, completed step]**; joint/body poses describe the
integrated endpoint. No instantaneous physical torque sensor is implied.

The geometry adapter runs `mj_kinematics` on a separate CPU `MjData` copy of the
actual solved coordinates. It checks Newton/native body-frame agreement and
covers every moving arm/tool collision geom, including rigid tool children.
Only the fixed mount/table and separate nut are excluded from the arm/tool
bounds. `geom_aabb` stores local center and **half extents**: world bounds use
`R*center + translation ± abs(R)*half_extent`, without an extra factor of 0.5.
The source identity must subsequently bind the complete model/registry and
reject model mutation; this checkpoint does not mistake a claimed digest for
that owner-side check.

Validation: 210 CPU checks passed in 0.96 s (105 new cases, existing contact and
fastening contract regressions); 660 source hashes and protected memory were
unchanged. A separate MuJoCo **3.12.0** box/rotation control passed in 0.016 s
with zero dynamics steps and no Newton/Warp/Kit import. The first invocation of
that control used the converter interpreter and failed before any model load
because its minimal environment lacks PyYAML. Reusing the task test interpreter
with the existing 3.12 packages read-only resolved that dependency without any
installation. The ordinary test environment uses MuJoCo 3.14 for its CPU FK
tests; this distinction is retained in the receipt. Independent effort-channel
controls also demonstrate why requested control, net actuator force, direct
applied force, passive force and constraint reaction must remain separate.
The first 159 checks passed before independent review found a missing-channel
edge: `np.any` on an empty wrench array looked like known zero. Exact float32,
shape and finite checks now precede any zero test; 39 added adversaries cover
empty, transposed and wrong-type buffers. No physical limit changed.
Further adversaries cover missing/invalid overflow counters, incomplete or
geometry-inconsistent body mappings and malformed arm quaternions. A missing
mapping or an invalid quaternion cannot silently bypass the FK comparison.
See [readback receipt](../benchmark/results/factory_observation_20261003.json).

Remaining implementation proceeds to the single native solve owner: model/asset
binding, upload-generation stamps, guarded target FK, exact zero-spindle upload
on every stop/fault/close path, immutable all-solve journal and bounded lifecycle.
That owner must additionally cross-check the real SDK `solver._step` and `d.time`;
the scene's caller-maintained counters alone do not establish a native solve.
Then the ordinary configuration builder can expose an explicit optional domain
through the existing MCP server. No active profile, controller launch, native
one-turn/rest verdict or physical admission is added by this readback checkpoint.

### Single solve owner checkpoint

`sim/factory_model.py` and `sim/factory_owner.py` now connect that readback to
the existing mounted `SeatingScene`. Construction does not start the owner.
`FactoryBoundModel` inventories the compiled MuJoCo model, actual Newton
collision arrays, native actuator/body/constraint parameters, options and
mapping tables, together with pinned source and mesh files. Imported joint,
control and effort limits are intersected explicitly. CPU shadow FK validates
the authored initial arm state before any solve. Mesh/SDF construction remains
source/asset-bound; the private owner exposes no model/SDF mutation API. This
is not an independent attestation of arbitrary external GPU memory mutation.

The owner bypasses the legacy scene's multi-step command loop. Each cycle
adopts at most one bounded admission request, checks the current solved state,
builds a measured-height follower target, prepares collision/coverage, checks
the final joint rate/tracking/FK bounds, uploads once and solves once. It checks the native
`solver._step` and exact float32 `d.time += dt` recurrence before/after the solve
and after reading it. Raw accumulated native time is retained separately from
the exact interval index times timestep. CUDA graph replay is unsupported by
this counter contract. No reader runs another solve or force evaluation.

Upload stamps capture generation, preceding step, original host times and
requested effort under the write fence. A stop during a solve leaves that
solve in its old upload generation. Only a subsequent zero-spindle upload can
produce the stop generation. A stop during target planning prevents the old
proposal from writing; it preserves the stop ACK's generation. Admission and
reset are processed between solves, so no earlier in-flight solve is relabeled
as new command evidence. The initial latch still requires fresh quiet measured
state and explicit `reset_stop`; a task boundary never resets it implicitly.
The final guarded upload defines admission of that one solve: no expensive
preparation follows this fence. A later stop does not promise to abort a solve
already admitted to the SDK. Review found the first implementation uploaded
before collision preparation; the retained synthetic causal control shows
0.03 Nm reaching the next solve after stop in that version, versus exact zero
in the new stop generation after moving the upload fence.

Stopped, faulted and closing paths submit exact zero to the spindle entry and
retain all previously approved arm targets. Upload failure remains visible;
neither an ACK nor zero command establishes physical rest. The passive journal
and detached raw queue cannot block on logger IO. Queue exhaustion is a sticky
producer failure. Closure joins the owner for at most two seconds and reports
an unclosed thread or failed zero upload. A blocked SDK call still requires an
external process watchdog: Python cannot interrupt an in-flight GPU operation.

The earlier `1e-7 Nm` observed-effort allowance is tightened in this checkpoint.
The native float32 channel must satisfy `abs(raw) <= max(cap, float32(cap))`;
requested commands still satisfy the exact original cap. For the spindle,
`float32(0.05) = 0.05000000074505806 Nm` is accepted and the next representable
float32 value is rejected. Raw measurements are never clipped or rounded.
The adapter validates float32 dtype and the explicit one-actuator/unit-gear
mapping before these observed limits apply.

Validation: 255 CPU checks passed in 1.16 s, including 45 owner/mapping/clock
adversaries; all 663 source hashes and protected stores stayed unchanged.
The tests use synthetic SDK buffers and CPU-only MuJoCo model serialization,
not native Factory execution. See [owner receipt](../benchmark/results/factory_owner_20261003.json).
The next checkpoint wires the optional domain into ordinary configuration and
the existing MCP server. A coordinated native campaign must still validate the
actual model construction, contact-force conversion, complete per-solve stream,
one turn, zero upload, observed rest and owned closure. Historical standalone
seating evidence supplies none of these new runtime verdicts.

### Configured runtime and existing MCP route

`configs/robots/factory_m20_mounted.yaml` declares `kind: fastening`, the fixed
recipe and official installer asset paths. Its `device: null` and
`model_identity_sha256: null` leave both choices unresolved and prevent native
construction before any SDK import. The operator's private preparation/profile
must select an explicit `cuda:N` device; `auto` and CPU are unsupported, with no
fallback. `load_robot_config` and `describe_robot` expose the namespaced
`fastening.turn_screw` catalog without opening a device. The normal
`build_robot_runtime` path creates this domain through
`apps/factory_runtime.py`; the existing MCP server selects that same composition
with `CASCADE_ROBOT=factory_m20_mounted`. There is no alternate server or fake
ArmBase. Discovery is not permission to actuate. Dynamic roots, any internal
multi-DoF joint, conflicting writers and unsupported profile fields are refused.

The paths match `fetch_factory_assets.py` (`assets/factory/nut_bolt`) and
`fetch_robot_assets.py so101` (`assets/mjcf/so101/so101.xml`). Model preparation
never fetches assets or silently substitutes SDK implementations. An explicitly
owned process may call `prepare_factory_model(profile, cache_dir)` to construct
the model, bind its actual sources/assets/mappings and obtain the exact digest
without solves or control uploads. A later fresh process must reproduce that
digest from a private pinned profile before starting the owner. The shipped
profile remains unpinned; a matching digest is not a physical success flag.

The configured builder records profile and model, refuses reused output epochs,
then starts the owner with its initial latch on and generation zero. The
readiness observer begins immediately after `owner.start()` and allows 10 host
seconds for 0.5 continuous physical seconds of measured rest with actual loaded
thread/tool engagement and exact zero spindle effort. Model construction and
compilation occur before that observer budget and must be timed separately by
the external harness. The owner's independent 120-host-second lifetime starts
at `owner.start()`. No automatic reset or motion occurs during readiness.
Neither these limits nor the existing turn/rest gates were extended.

#### Explicit initial-posture variant (2026-10-03, CPU authoring only)

Preparation V3 on `8f7f5b37128419fa10ef5a8b93fa6a9fad7e0373` failed
before a valid model pin: the legacy initial `wrist_flex` was
`1.6446820497512817` rad after float32 storage. Its joint/control intersection
ends at `1.65806` rad, leaving about `0.013378` rad, below the unchanged
`0.02` rad guard. All other controlled joints passed this CPU comparison.
The failed native process measured scene/native step and time zero and closed
naturally with exit 1; it did not run an owner or readiness. The full failure and
closure remain in `benchmark/results/factory_initial_margin_20261003.json`.

`factory_m20_mounted` and the default `SeatingScene` preserve the legacy recipe
and physical placement. The separate profile
`factory_m20_mounted_margin_v2` selects `factory_m20_fixed_axis_margin_v2`:
move the **whole static bolt, pre-engaged nut and seat ring** 10 mm toward the
base to `(0.23, 0)` m; keep their heights, thread geometry and mounted socket
unchanged. Solve the arm IK within the actual joint/position-control intersection
with `0.025` rad clearance. There is no joint clipping, gain/torque change,
residual relaxation or reuse of earlier physical success. Both profiles still
ship with null device and model pin.

Selection used only pinned CPU kinematics. With the same vertical socket target,
`x=0.24`, `0.25` and `0.26` m failed the stricter IK residual bound; `x=0.23` m
passed all 43 heights from 0.069 to 0.027 m. Minimum joint clearance was
`0.025004558855823644` rad; maximum weighted residual was
`2.107337996387422e-10`, against the existing `1e-5` criterion. The actual Newton
1.6 importer and CPU FK independently retained zero reference offsets for all
six controlled joints, and the 43 float32 targets had maximum weighted residual
`1.0687176054905729e-7`. This control constructed no Factory SDF, collision
pipeline or physics solver and took no dynamics steps. The first CPU importer
invocation lacked `trimesh`; its failure is retained separately from the second
invocation with the declared read-only mesh dependency.

The model document binds the consumed recipe, center, socket offset, IK margin,
range intersection, entry and all follower targets. Binding checks the variant
parameters and every authored target; immutable checks also reject later target
table changes. Initial rejection now includes joint name, q, bounds and margin.
This new geometry/identity still needs a separately reviewed native preparation,
fresh readiness and task measurement. Kinematic reachability does not establish
thread/tool contact, collision freedom during motion, seating or physical stop.

Preparation V4 on `96ecdb59650f9943a4e3c09c030c443af4d88b92` reached
the binding constructor but rejected `fixture_origin_m`: converting the native
NumPy array to a tuple retained `numpy.float64` elements, while the contract
requires Python numeric scalars. Its observed scene/native steps and times were
zero; the process closed naturally with exit 1 and no model pin. The correction
converts each already-validated coordinate explicitly to Python `float` at that
boundary. The strict public contract, coordinate values and physical criteria
are unchanged. The exact constructor assignment reproduced both recipe failures
on CPU; after correction, 362 focused tests passed in 1.48 s with source/store
hashes unchanged. This is a binding fix, not a successful native preparation.
Raw failure, closure and causal controls are linked in
`benchmark/results/factory_binding_scalars_20261003.json`.

Detached all-solve records, readiness, action and closure receipts are persisted
outside the priority stop/write path. A persistence failure revokes authority,
leaves the evidence fault sticky, refuses reset and makes any task result
unverified while preserving its execution result. Startup preserves its original
exception if closure or writing the failure receipt also fails; the exception
notes retain the causal record when the output device is unavailable. A failed
closure is never promoted to success. A process watchdog remains necessary for
SDK construction or an uninterruptible solve; a Python join cannot enforce a
GPU deadline.

Validation: 322 CPU tests passed in 4.40 s across seven configuration, MCP,
composition, fastening, owner, readback and generalized-joint files. Thirty
configuration cases then passed in 0.35 s after correcting only the two profile
asset paths to the official installers' layout. Each run preserved all 666
source/config/test hashes and protected stores. Initial failed fixture controls
are retained: a startup error-message branch and stale resource references in a
synthetic embodiment, subsequently corrected without weakening any gate.
Independent read-only review covered startup/closure failure persistence and
confirmed the final two path changes; it did not repeat tests or native physics.
See the [configuration receipt](../benchmark/results/factory_config_20261003.json).

The remaining admission is a new coordinated native campaign: exact SDK/model
construction, contact-force conversion and complete solve stream, unchanged
quiet-start limits, one measured turn, zero upload, observed rest, same-episode
visual evidence and owned closure. The historical standalone seating result
does not supply any of these new driver's verdicts. No pickup, automatic
engagement, seat or preload capability is exposed.

A subsequent composed-suite check caught the shipped explicit CUDA ordinal
against CASCADE's existing device-neutral configuration policy. The profile now
leaves `device: null`; private native preparation must still choose `cuda:N`
explicitly. Fifty CPU tests passed in 0.44 s, including the unchanged global
policy test and refusal before SDK/output creation. The first local edit's YAML
indentation failure is retained separately. No CPU fallback or physical limit
change was introduced. See the [device-selection receipt](../benchmark/results/factory_device_selection_20261003.json).

### First construction and pinned SDK interface correction

Source `4e53a8a` reached scene construction in a single prepare-only process on
an explicitly shared GPU, then failed before binding: Newton 1.6 exposes
`SolverMuJoCo._use_mujoco_contacts`, not the public attribute used by the new
adapter. The process exited naturally with code 1; its owned scope and births
were absent afterward, all source/SDK/assets/store hashes were unchanged, and
the six pre-existing GPU process identities were unchanged. No owner, readiness,
reset or motion was requested. The failed return supplies neither a model pin
nor measured zero-solve counters. The 31.876 s construction interval under shared
load is a diagnostic duration, not a performance result.

The adapter now requires the pinned stored mode and the effective MJWarp
`opt.run_collision_detection` branch to be explicitly false, together with
`use_mujoco_cpu`. Missing fields, wrong types, contradictions and later changes
reject; there is no alternate-field fallback. The fingerprint binds these modes.
The native failure remains intact; this correction has not been run natively.

Sixteen new CPU controls produced 15 failures and one pass on the old code;
the corrected five-file suite passed 274 tests in 1.36 s. A separate 163-test
owner/readback run passed with the exact MuJoCo 3.12 package in 0.45 s. Static
inspection found the fingerprint's 33 Newton arrays, six maps, 43 MJWarp arrays
and 16 option fields declared in the pinned SDK, and checked the owner method
signatures and step counter. These checks do not establish native buffer
coverage, readiness, physical task success or a valid model pin. See the
[SDK-interface receipt](../benchmark/results/factory_sdk_interface_20261003.json).

Preparation V2 on `af463859` passed that mode check but failed while fingerprinting
a Newton array whose value was `None`. The exception diagnostic measured scene
and native counters at step 0 / time 0 in epoch `d77aab0af99a4a3ebed35fdee401deee`.
The process again exited 1 naturally, with no owner or action and no model pin;
all inputs and the six peer births were unchanged. Its construction interval
was 8.8466 s under shared load. Neither duration is a performance comparison.

A minimal real Newton 1.6 CPU builder isolated `shape_filter` as the only absent
field among the 33 fingerprint arrays and reproduced the production error.
The SDK declares that optional field but its builder does not populate it.
Its absence is now represented explicitly as `present: false` with declared
`int32[shape_count]` layout. If present, the exact layout is required. Missing
attributes and every other required array still reject; no absence is converted
to an empty array. Presence transitions and changes to shape, dtype or values
invalidate the identity.

The corrected CPU suite passed 375 tests in 1.56 s. A second CPU construction
passed the new Newton fingerprint and recorded actual layouts for all 43
selected MJWarp arrays and seven option arrays using `put_model`, without a
solver or simulation step. The SDK/asset/harness/protected-store inventories
still match the closed V2 run. Static review also records when actuator mappings
can be absent: only without mapped actuators, which this fixture already refuses.
These CPU observations do not instantiate or admit the full Factory fixture.
See the [optional-array receipt](../benchmark/results/factory_optional_array_20261003.json).
