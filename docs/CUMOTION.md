# Optional cuMotion motion planning

Cascade provides a standalone adapter for NVIDIA **cuMotion 1.1.0** through
`cascade.planning.make_motion_planner`. It generates joint-space trajectory
candidates using the SDK's `TrajectoryOptimizer` and a caller-supplied static
world. The demo, MCP tools, grasp selector and `SafeArm` retain their existing
planning and execution paths.

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
These are upstream platform claims. The native smoke below establishes one
Linux x86_64 GPU planning result; ARM, Windows and physical task execution
remain unvalidated in Cascade.

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

Runtime execution integration requires a separate trajectory-aware SafeArm
path retaining its existing stop, velocity, physical-clock, current scene,
contact/payload and recovery gates, plus independent physical validation.
There is no claim that cuMotion improves the current kitchen campaign or has
completed hardware/Isaac acceptance.

## Validation

On source `81b2070` (merged in [PR #47](https://github.com/johnnynunez/cascade/pull/47)),
the actual cuMotion 1.1.0 CUDA 13.0 wheel passed the public CLI/factory on Linux
x86_64 with an RTX PRO 6000 Blackwell. One small Franka request with a distant
static cuboid produced a 0.219-second trajectory with 12 samples in 0.732 wall
seconds. Maximum endpoint error was 1.15e-7 rad. Only the selected GPU was
observed, peaking at 582 MiB; the child exited normally and source/model hashes
and existing process identities were unchanged. The
[compact receipt](../benchmark/results/cumotion-planner-20261001.json) binds the
candidate, supervisor, model metadata and independent review.

This is one native integration smoke. It does not measure ARM support,
continuous/live-world collision clearance, robot execution or kitchen success.
The SDK was installed in a separate environment; no presenter environment or
runtime source changed. Model parsing exposed stale frame names in an upstream
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
