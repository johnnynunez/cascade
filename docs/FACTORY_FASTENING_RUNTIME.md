# Mounted Factory fastening runtime

Stage one, 2026-10-03. This is an optional domain implementation with CPU contract
tests. There is no shipped physical profile or new native result yet. The
[ordinary wrist routine](FASTENING_RUNTIME_GAP.md) remains unverified.

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
