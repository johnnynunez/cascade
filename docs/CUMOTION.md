# Optional cuMotion motion planning

Cascade provides an optional adapter for NVIDIA **cuMotion 1.1.0** through
`cascade.planning.make_motion_planner`. It generates joint-space trajectory
candidates using the SDK's `TrajectoryOptimizer` and a caller-supplied static
world. The explicit `isaac_cumotion` arm profile now connects it to normal
demo/MCP manipulation through `SafeArm`; other profiles retain their existing
execution path.

This adapter is implemented against the [official Python API](https://nvidia-isaac.github.io/cumotion/api/python_api.html)
and [trajectory optimization tutorial](https://nvidia-isaac.github.io/cumotion/tutorials/trajectory_optimization_tutorial.html).
The upstream reference is [cuMotion v1.1.0](https://github.com/nvidia-isaac/cumotion/releases/tag/v1.1.0),
source commit `ac115fdb7737a4da07251ea85d15e4063be1ce6e`.

## Install the optional SDK

cuMotion is distributed separately under the
[NVIDIA Isaac ROS Software License](https://github.com/nvidia-isaac/cumotion/blob/ac115fdb7737a4da07251ea85d15e4063be1ce6e/LICENSE).
Cascade does not redistribute its binaries, headers or robot models and does
not install it as a base dependency. Its use is subject to NVIDIA's terms.

The [upstream requirements](https://github.com/nvidia-isaac/cumotion#system-requirements)
list Python 3.10–3.14 wheels, Linux x86_64 with CUDA 12.6/13.0, aarch64 on
Jetson Orin with CUDA 12.6 and Jetson Thor/DGX Spark with CUDA 13.0, and
Windows x86_64 with CUDA 13.0. x86 requires an NVIDIA Turing GPU or newer.
These are upstream platform claims. Cascade has native GPU planning smokes on
Linux x86_64 and DGX Spark aarch64. Windows, other ARM devices and hardware
task execution remain unvalidated in Cascade. The Isaac validation scope is
recorded below.

Download the matching release archive and follow
[NVIDIA's installation instructions](https://nvidia-isaac.github.io/cumotion/getting_started.html).
Install the wheel in a separate environment with Cascade's base dependencies;
keep any existing presenter, Isaac or nvblox environment unchanged. The adapter
requires the installed distribution to report version `1.1.0`; another version,
missing binary dependency or failed CUDA initialization raises `PlanningError`.
There is no fallback to another solver. Select the GPU with
`CUDA_VISIBLE_DEVICES` **before** starting the process.

## Configure and export a candidate

Copy [the example profile](../configs/planners/cumotion.example.yaml), set its
URDF and XRDF paths, and supply the model's exact base/tool frames and controlled
joint names. The XRDF must include world and self collision spheres. Both files
are read once and their actual contents are hashed before passing them to
`load_robot_from_memory`.

The 1.1.0 CUDA 13.0 Linux x86_64 release's bundled `franka.urdf` and
`franka.xrdf` report `base_link` and `panda_leftfingertip`. The example profile
uses those measured frame names; verify the names again for another model or
SDK archive. Some upstream tutorial examples still name `right_gripper`, which
is not a declared tool frame in this bundled XRDF.

For the upstream Franka example, a small joint-space request looks like:

```bash
python -m cascade.apps.plan_motion --config /path/to/planner.yaml \
  --start 0 -0.5 0 -2 0 1.5 0 \
  --goal 0.1 -0.5 0 -2 0 1.5 0 \
  --output /path/to/new-candidate.json
```

The CLI never opens an arm or simulator connection and refuses to overwrite an
existing output. Native CUDA work is synchronous; `max_duration_s` bounds the
*returned trajectory duration*, not solve wall time. Use a process supervisor
when a hard wall-time deadline is required. NVIDIA's native library can report
fatal model/CUDA errors at process level; run unqualified models in a separate
process.

The same API accepts a `dict` or Cascade `Cfg`:

```python
from cascade.planning import make_motion_planner

with make_motion_planner(config) as planner:
    candidate = planner.plan(start_q, goal_q)
    record = candidate.as_dict()
    assert record["execution_authorized"] is False
```

All joint vectors use the caller's `joint_names` order. `joint_signs` explicitly
maps each local coordinate to the URDF coordinate; the adapter reorders by
XRDF joint names and applies the inverse mapping to positions and velocities.
It does not assume that URDF declaration order equals actuator order or
automatically adopt a Cascade arm profile. Positions and obstacle dimensions
are metres, revolute joints are radians, and times are seconds.

`obstacles` is required. An explicit empty list means an empty external world.
Each cuboid uses full dimensions and a rigid transform from its centre frame
to the robot base frame, for example:

```yaml
obstacles:
  - name: table
    size_m: [1.0, 1.0, 0.05]
    T_base_box:
      - [1, 0, 0, 0.5]
      - [0, 1, 0, 0]
      - [0, 0, 1, -0.025]
      - [0, 0, 0, 1]
```

Geometry is copied at construction and the SDK world view is updated before
creating the solver. A planner instance has one immutable world binding;
construct another instance when the scene changes. Unknown fields, unsupported
geometry, duplicate names, invalid transforms and mismatched model frames or
joints fail explicitly. Both SDK collision checks are enabled and their
configuration results are checked.

## Result and safety boundary

The adapter accepts only native `SUCCESS`, validates the trajectory's domain,
coordinate count, finite positions/velocities, joint limits and margin, global
position/velocity extrema, endpoints, sample count and inter-sample velocity
bounds. It also checks each sample with the SDK's world/self collision
inspector. A failed or malformed native result faults that instance; there is
no automatic retry. The context manager releases the solver before its world
and robot owners.

The returned immutable samples include model, scene and request hashes.
`status: candidate` and `execution_authorized: false` are deliberate: sampled
collision checks against XRDF spheres/static cuboids are not a proof of
continuous clearance, current world geometry or actuator tracking. The adapter
does not consume live nvblox, observed-finger, held-object, attachment or release
evidence. It does not apply velocities, efforts, gripper commands or resets.
Passing these samples one by one to `move_joints` would create different
interpolation curves and does not preserve the planned path.

## Normal Isaac manipulation

Select `isaac_cumotion` as the arm profile in the normal demo or MCP runtime.
It extends `isaac`; use the same camera calibration, bridge address and task
configuration as the Isaac deployment. Enable the observed-finger gate with
`CASCADE_OBSERVED_FINGER_GATE=1` for the validated contact path; retain real
required GraspGenX and the deployment's occupancy policy. The separate SDK must
be importable in that application's interpreter. Missing SDK/model/coordinate bindings fail
before the lazy arm is materialized. Other arm profiles do not import cuMotion.

The shipped reBot [XRDF and provenance](../assets/cumotion/rebot/PROVENANCE.md)
are bound to the exact arm URDF, controlled joint names/order, mirrored signs,
root and `gripper_end` frame. Static finger spheres are an auxiliary SDK model;
they do not represent measured jaw strokes, a live scene or a held payload.
`obstacles: []` explicitly adds no static obstacles. Existing table, workspace,
observed-finger and configured occupancy/payload vetoes retain authority.

Ordinary `SafeArm.move_joints` and `move_planned` requests obtain one native
curve, uniformly slow it to the requested duration/host velocity bound, copy
its original values at both the 30 Hz command grid and existing 50 Hz safety
grid, and validate that exact curve before streaming. There is no per-sample
minimum-jerk reconstruction. Physical-clock/epoch binding, unchanged start
feedback, live approvals, stop generation, watchdog, contact/attachment and
release callbacks remain enforced. Native failure, budget expiry, drift or a
veto is terminal; none selects another planner or interpolator.

Grasp descent, initial lift and an allowed pre-close withdrawal explicitly
request native linear translation and constant-orientation constraints. A
separate native contact optimizer is initialized from a direct joint path
(`trajopt/pbo/enabled=false`), with documented PBO/L-BFGS path-position weights
of 600000/50000000. It requests a 0.1 mm corridor and independently rejects any
copied command or safety sample more than 1 mm from the **closed** segment
between the measured starting TCP and goal, or more than 0.025 rad from the
terminal orientation. After closing, a bounded observation requires at least
0.5 seconds of advancing physics with at most 0.5 mrad joint range. The
measured stable orientation feeds lift IK at the vetted pregrasp position;
independent FK must agree within 0.1 mm/0.1 mrad. Existing halt/attachment
authority remains active throughout this wait, and later planning still
requires the same 1 mrad start-feedback bound. Free-space transit retains the
normal native optimizer.
This policy is selected before solving; an invalid curve is never replaced.

SDK initialization occurs before actuator creation. A runtime solve has the
existing 3-second planning budget, checked before actuation; synchronous native
work cannot be preempted in-process. The profile defaults to a 0.025 rad joint
margin. An explicit per-motion harness margin is propagated into all native
endpoint/extrema/sample checks and the request hash; hard URDF limits and
collision checks still apply. The folded mechanical zero self-collides in the
shipped sphere model, so this Isaac-only profile declares the validated home
configuration as `park_q`. Shutdown skips motion when a held/provisional
object or unresolved contact, attachment or release is retained. Disconnecting
the Isaac client leaves the simulator's articulation drives active.

## Physical runtime validation — 2026-10-02

The normal `pick_and_place("green object", "drop zone")` task completed in an
isolated Isaac tabletop scene using cuMotion for every arm movement and OVRTX
for the real RGB-D/mask observations. The application source was
`f2b2186317a6eb6278bf7a57bf039c912e64dc18`; recorded source hashes remained
unchanged through execution and shutdown. The run used PhysX CUDA, CUDA
perception and required GraspGenX on Linux x86_64 with the CUDA 13.0 cuMotion
1.1.0 wheel and an RTX PRO 6000 Blackwell. The observed-finger gate was enabled;
occupancy was explicitly disabled under the supported isolated runtime policy.

Eight task curves delivered 529 acknowledged targets, including native
constrained descent and lift, with 2,545 replayed safety edges. One additional
native home-park curve completed during shutdown. Native planning took
0.789–1.035 wall seconds per request in this run; these figures are evidence of
the configured budget, not a performance comparison. The task verified its
grip, transported the object 0.3325 m, released it 23.5 mm from the configured
drop point and returned home.

Release acceptance did not depend only on the skill's result flag: 24 fresh
physics samples over 1.283 seconds showed open jaws and no object position
change, with its centre at the nominal support height of 0.040 m. Twelve more
samples after shutdown park confirmed the same pose and no attachment. This
is pose/height evidence of support and settling, not a measured table reaction
force. The [compact cuMotion receipt](../benchmark/results/cumotion-runtime-20261002.json)
records the native requests, source hashes, limits and raw artifact hashes;
the [OVRTX receipt](../benchmark/results/ovrtx-runtime-manipulation-20261002.json)
records camera/physics binding and process ownership. See also the
[before/after figure](../benchmark/results/images/ovrtx-runtime-green-20261002.png).

The receipts preserve failures as well as the accepted run:

| Trial | Measured outcome |
| --- | --- |
| Pink 02 | Unconstrained descent rejected by observed-finger geometry before close. |
| Pink 03 | **Invalid for observed-finger acceptance:** the runner omitted the required environment flag. An independent closed-segment path check rejected descent; subsequent runners assert the flag before creating the runtime. |
| Pink 04 | Guarded descent and close passed; lift failed the unchanged 0.025 rad orientation bound. Lift IK now preserves the measured post-close orientation. |
| Pink 05 | Contact passed, but start drift after closing exceeded the unchanged 1 mrad streaming bound. The runtime now observes bounded physical settling before planning. |
| Pink 06 | Fresh observed geometry vetoed a diagonal empty withdrawal; a separately admitted 4 cm vertical withdrawal and home completed. The old folded park pose also failed SDK self-collision and was replaced by the explicit home park configuration. |
| Pink 07 | Native constrained lift raised the object 40.5 mm, then joint 6 failed the unchanged 0.045 rad settling bound with 0.048762 rad following error. The possible payload was retained and shutdown correctly skipped parking. |

Pink 07's native endpoint matched the live articulation target with the
expected mirrored joint sign. The observed joint-6 drive used stiffness
2864.789, damping 401.070 and a 14 Nm effort limit. These observations locate
the error in physical following rather than coordinate conversion; they do
not establish a unique controller/contact root cause. No gains or settling
tolerances were changed. Only that owned simulator was terminated after
checking its process birth identities, without opening the raised gripper or
writing object poses. The OVRTX receipt separately retains its guarded pink
contact rejection and the earlier green post-close stability failure.

The final post-close prefilter allows 0.5 mrad range over 0.5 advancing physics
seconds, reserving half of the existing 1 mrad start-drift allowance. The
initial newly introduced 0.25 mrad prefilter rejected measured 0.356 mrad
contact jitter; correcting that policy did not change final trajectory
admission or the wall deadline. Fresh measured state, the original halt
generation and retained attachment authority remain required.

The focused CPU suite passed 265 tests covering planner conversion, original
curve execution, physical clocks, cancellation, per-call margins, contact
stability, grasp/carry behavior and guarded shutdown. These tests use SDK
doubles; the native run above supplies the separate physical evidence. The
accepted scope is one complete simulated tabletop task. It does not establish
a success rate, hardware execution, kitchen acceptance, live nvblox payload
geometry or continuous swept-volume clearance.

## Earlier adapter-only validation

On source `81b2070` (merged in [PR #47](https://github.com/johnnynunez/cascade/pull/47)),
the actual cuMotion 1.1.0 CUDA 13.0 wheel passed the public CLI/factory on Linux
x86_64 with an RTX PRO 6000 Blackwell. One small Franka request with a distant
static cuboid produced a 0.219-second trajectory with 12 samples in 0.732 wall
seconds. Maximum endpoint error was 1.15e-7 rad. Only the selected GPU was
observed, peaking at 582 MiB; the child exited normally and source/model hashes
and existing process identities were unchanged. The
[compact receipt](../benchmark/results/cumotion-planner-20261001.json) binds the
candidate, supervisor, model metadata and independent review.

The same source and model also passed the real CUDA 13.0 aarch64 wheel on
DGX Spark GB10: 12 samples over 0.219 seconds, generated in 0.703 wall seconds,
with the same endpoint error. The selected child peaked at 200 MiB of reported
GPU process memory and exited normally. All 2,007 isolated source/environment
files, 1,083 active runtime files and 11 existing process identities remained
unchanged. GPU total memory was unavailable; two graphics-process working
directories were unreadable and are recorded as such.

These are individual native integration smokes, not a comparative performance
benchmark or evidence of continuous/live-world clearance, robot execution or
kitchen success. Each SDK was installed in a separate environment; no presenter
environment or runtime source changed. Model parsing exposed stale frame names in an upstream
example, which the shipped profile now corrects.


`tests/test_cumotion_planner.py` uses a CPU SDK double matching the documented
1.1.0 interfaces. It exercises joint order/sign conversion, nonzero trajectory
time origins, static obstacle binding, malformed native results, failures,
model/scene hashes, input immutability, lazy imports and the public CLI.
These tests validate Cascade's adapter contract; they are not native SDK or
GPU measurements.

The implementation uses these documented 1.1.0 calls directly:

| Adapter responsibility | Upstream Python interface |
| --- | --- |
| Exact model contents | `load_robot_from_memory(xrdf, urdf)` |
| Coordinate and frame binding | `RobotDescription.cspace_coord_name`, `tool_frame_names`, `kinematics` |
| Position/velocity limits | `Kinematics.cspace_coord_limits`, `cspace_coord_velocity_limit`, `base_frame_name` |
| Static cuboids | `create_obstacle`, `Obstacle.Attribute.SIDE_LENGTHS`, `Pose3(matrix)`, `World.add_obstacle` |
| Published static world | `World.add_world_view`, `WorldViewHandle.update` |
| Collision coverage and samples | `create_robot_world_inspector`, `num_world_collision_spheres`, `num_self_collision_spheres`, `in_self_collision`, `in_collision_with_obstacle` |
| Planner construction | `create_default_trajectory_optimizer_config`, `set_param`, `create_trajectory_optimizer` |
| Joint-space request | `TrajectoryOptimizer.CSpaceTarget`, `plan_to_cspace_target` |
| Result admission | `Results.status`, `Results.Status.SUCCESS`, `Results.trajectory` |
| Trajectory inspection | `Trajectory.domain` (`lower`/`upper`), `num_cspace_coords`, `eval(t, derivative_order)`, `min_position`, `max_position`, `max_velocity_magnitude` |
