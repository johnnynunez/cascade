# Optional cuMotion motion planning

**Next action:** export one candidate trajectory from the Franka example (seconds once the SDK
env exists; `/path/to/cumotion-env` is the separate environment from step 1 below):

```bash
/path/to/cumotion-env/bin/python -m cascade.apps.plan_motion --config /path/to/planner.yaml \
  --start 0 -0.5 0 -2 0 1.5 0 \
  --goal 0.1 -0.5 0 -2 0 1.5 0 \
  --output /path/to/new-candidate.json
```

The CLI never opens an arm or simulator connection and refuses to overwrite an existing output.
The adapter (`cascade.planning.make_motion_planner`) wraps NVIDIA **cuMotion 1.1.0**'s
`TrajectoryOptimizer` with a caller-supplied static world
([Python API](https://nvidia-isaac.github.io/cumotion/api/python_api.html),
[trajectory optimization tutorial](https://nvidia-isaac.github.io/cumotion/tutorials/trajectory_optimization_tutorial.html),
[v1.1.0](https://github.com/nvidia-isaac/cumotion/releases/tag/v1.1.0) at
`ac115fdb7737a4da07251ea85d15e4063be1ce6e`). The `isaac_cumotion` and `isaac_kitchen_cumotion` arm
profiles connect it to normal demo/MCP manipulation through `SafeArm`; other profiles never import
it.

Status:

1. **Works (simulation):** one complete Isaac tabletop `pick_and_place` with cuMotion for every arm
   movement; native planning smokes on Linux x86_64 and DGX Spark aarch64.
2. **Not shown:** a success rate, kitchen acceptance (`isaac_kitchen_cumotion` has none), Newton,
   hardware, live nvblox payload geometry, continuous swept-volume clearance; Windows and other ARM
   devices are unvalidated in Cascade.

## Install and configure

1. cuMotion is distributed separately under the
   [NVIDIA Isaac ROS Software License](https://github.com/nvidia-isaac/cumotion/blob/ac115fdb7737a4da07251ea85d15e4063be1ce6e/LICENSE);
   Cascade does not redistribute or install it. Upstream lists Python 3.10–3.14 wheels for Linux
   x86_64 (CUDA 12.6/13.0, Turing or newer), Jetson Orin (12.6), Jetson Thor/DGX Spark (13.0) and
   Windows x86_64 (13.0). Follow [NVIDIA's installation](https://nvidia-isaac.github.io/cumotion/getting_started.html)
   into a **separate environment** with Cascade's base dependencies; leave presenter, Isaac and
   nvblox environments unchanged.
2. The installed distribution must report `1.1.0`; another version, a missing binary dependency or
   failed CUDA init raises `PlanningError` (no fallback solver). Set `CUDA_VISIBLE_DEVICES`
   **before** starting the process.
3. Copy [the example profile](../configs/planners/cumotion.example.yaml), set URDF and XRDF paths
   (the XRDF needs world and self collision spheres), the model's exact base/tool frames and
   controlled joint names. Both files are read once and hashed before `load_robot_from_memory`.
   The bundled 1.1.0 `franka.urdf`/`franka.xrdf` report `base_link` and `panda_leftfingertip`
   (some upstream tutorials still name `right_gripper`, which this XRDF does not declare).
4. Joint vectors use the caller's `joint_names` order; `joint_signs` maps each local coordinate to
   the URDF coordinate (no assumption that URDF order is actuator order). Units: metres, radians,
   seconds.
5. `obstacles` is required (`[]` means an empty world). Cuboids use full dimensions and a rigid
   transform from their centre to the robot base:

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

One planner instance has one immutable world binding; construct another when the scene changes.
Unknown fields, unsupported geometry, duplicate names, invalid transforms and mismatched frames or
joints fail explicitly. The same API accepts a `dict` or Cascade `Cfg`:

```python
from cascade.planning import make_motion_planner

with make_motion_planner(config) as planner:
    candidate = planner.plan(start_q, goal_q)
    record = candidate.as_dict()
    assert record["execution_authorized"] is False
```

## Result and safety boundary

1. Only native `SUCCESS` is accepted; the trajectory's domain, coordinate count, finite values,
   joint limits and margin, extrema, endpoints, sample count and inter-sample velocity are
   validated, and every sample goes through the SDK's world/self collision inspector. A malformed
   result faults the instance; no automatic retry.
2. Results are `status: candidate` and `execution_authorized: false`, with model, scene and request
   hashes: sampled checks against XRDF spheres and static cuboids do not prove continuous clearance,
   current world geometry or actuator tracking.
3. The adapter consumes no live nvblox, observed-finger, held-object, attachment or release evidence
   and applies no velocities, efforts, gripper commands or resets. Feeding samples one by one to
   `move_joints` would not preserve the planned path.
4. Native CUDA work is synchronous: `max_duration_s` bounds the returned trajectory duration, not
   solve wall time. Use a process supervisor for a hard deadline, and run unqualified models in a
   separate process (native errors can be fatal to the process).

## Normal Isaac manipulation (`isaac_cumotion`)

Select `isaac_cumotion` (extends `isaac`; same camera calibration, bridge and task configuration)
with the SDK importable in the application interpreter. Enable `CASCADE_OBSERVED_FINGER_GATE=1` for
the validated contact path and keep real required GraspGenX and the deployment's occupancy policy.
Missing SDK/model/coordinate bindings fail before the lazy arm is materialized.

1. The shipped reBot [XRDF and provenance](../assets/cumotion/rebot/PROVENANCE.md) bind the exact
   URDF, joint names/order, mirrored signs, root and `gripper_end` frame; static finger spheres are
   not measured jaw strokes; `obstacles: []` adds no static obstacles. Table, workspace,
   observed-finger and occupancy/payload vetoes keep authority.
2. Grasp selection checks pregrasp, grasp and carry-lift endpoints against the same model (limits,
   margin, self collision, static obstacles) before any command; inspector failure is terminal.
3. `SafeArm.move_joints` / `move_planned` get one native curve, slowed uniformly to the requested
   duration and copied onto the 30 Hz command and 50 Hz safety grids, then validated before
   streaming. Clock/epoch binding, start feedback, approvals, stop generation, watchdog and
   contact/release callbacks stay enforced; failure, budget expiry, drift or a veto is terminal
   (no other planner or interpolator).
4. Descent, initial lift and pre-close withdrawal request native linear translation with constant
   orientation (contact optimizer from a direct joint path, `trajopt/pbo/enabled=false`, PBO/L-BFGS
   weights 600000/50000000): copied samples more than 1 mm off the closed TCP segment or 0.025 rad
   off the terminal orientation are rejected. After closing, ≥ 0.5 s of advancing physics with
   ≤ 0.5 mrad joint range is required (half the 1 mrad start-drift bound); lift IK uses the measured
   orientation (FK agreement within 0.1 mm / 0.1 mrad).
5. A runtime solve has the 3 s planning budget, checked before actuation. Default joint margin 0.025
   rad (a per-motion harness margin propagates into every check and the request hash). The folded
   mechanical zero self-collides in the sphere model, so the profile declares the validated home as
   `park_q`. Shutdown skips motion while a held/provisional object or unresolved contact, attachment
   or release is retained; disconnecting leaves the simulator's drives active.

## Kitchen selection (`isaac_kitchen_cumotion`, no native kitchen acceptance)

It extends `isaac_kitchen_gpu` (grasp, placement, safety, occupancy, timing) with the same planner
recipe and park pose. Campaign, with the producer and SDK already configured:

```bash
python benchmark/diagnostics/kitchen_acceptance.py --port 8681 --engine physx \
  --arm-profile isaac_kitchen_cumotion --camera-renderer ovrtx \
  --objects green_cube --rounds 1 --fail-fast --output runs/kitchen-cumotion-01
```

Explicit Spark launcher selection (installer and default launcher keep `isaac_kitchen_gpu`):

```bash
CASCADE_INSTALL_PROFILE=spark ./scripts/launch.sh --sim isaac --engine physx \
  --arm isaac_kitchen_cumotion --camera-renderer isaac \
  --scene-config demo/scene/kitchen_config.json --occupancy none
```

1. The renderer argument is an expectation, not an override; the launcher requires it for this arm
   and rejects a conflicting inherited renderer. `ovrtx` also needs an existing
   `CASCADE_OVRTX_PYTHON` and a fresh `CASCADE_OVRTX_OUTPUT`.
2. `CASCADE_KITCHEN_CAMERA_RENDERER` carries the expectation into the builder, which rejects a
   different arm, multiple arms, a missing planner or non-strict CUDA before construction. Spark
   attestation, camera readiness, physical-clock execution and the 300,000 ms MCP deadline stay.
3. Each motion checks cached renderer packets and model/world/request-bound `MotionPlan` evidence
   (no extra camera/state RPCs). `context.selected_backends` keeps at most 128 curves, 256 checks and
   128 errors per call (overflow rejects further execution) with `physical_acceptance: false`.
4. Occupancy is a separate `--occupancy nvblox` selection; omitting it validates no mapping.
5. A run without a successful pick-phase native curve cannot claim selected-cuMotion acceptance;
   executor completion alone is not physical success.

## Evidence

1. **Tabletop task, 2026-10-02:** `pick_and_place("green object", "drop zone")` in an isolated
   Isaac scene, cuMotion for every movement, OVRTX RGB-D, PhysX CUDA, required GraspGenX, RTX PRO
   6000 Blackwell, source `f2b2186317a6eb6278bf7a57bf039c912e64dc18`: eight curves, 529 acknowledged
   targets, planning 0.789–1.035 s per request, object moved 0.3325 m and released 23.5 mm from the
   drop point; 24 post-release physics samples showed open jaws and no motion
   ([cuMotion receipt](../benchmark/results/cumotion-runtime-20261002.json),
   [OVRTX receipt](../benchmark/results/ovrtx-runtime-manipulation-20261002.json),
   [figure](../benchmark/results/images/ovrtx-runtime-green-20261002.png)).
2. **Retained failures (pink 02–07)** drove the current rules: orientation-preserving lift IK,
   bounded post-close settling, a vertical withdrawal, the home park pose; pink 07 still failed the
   unchanged 0.045 rad joint-6 settling bound (0.048762 rad) and no gains or tolerances were
   changed. Details in the receipts.
3. **Adapter smokes** (PR #47): Franka request, 12 samples over 0.219 s, endpoint error 1.15e-7 rad,
   0.732 s on x86_64 RTX and 0.703 s on DGX Spark GB10
   ([receipt](../benchmark/results/cumotion-planner-20261001.json)). Not a performance benchmark.
4. **CPU contract:** `tests/test_cumotion_planner.py` (SDK double matching the 1.1.0 interfaces);
   not a native or GPU measurement.

Upstream 1.1.0 calls the adapter uses:

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
