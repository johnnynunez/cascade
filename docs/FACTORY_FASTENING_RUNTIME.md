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
