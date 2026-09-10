# Design: mobility, humanoids and navigation (`MobileBase`)

Status: **design, not code** (2026-09-10). Approved direction: Unitree G1/H1
humanoid in simulation first; one `MobileBase` interface with two navigation
backends -- ROS2 Nav2 when a ROS2 environment is sourced, a Warp costmap +
planner otherwise -- so the one-click keeps working on a laptop. Code lands
in a follow-up after this design is reviewed.

## What exists today, precisely

- **Arms only.** `control/arm_base.py` (`ArmBase`, six methods) and every
  backend under `control/` drive an *arm*. `type: ros2` (`ros2_arm.py`) is
  `sensor_msgs/JointState` in, `trajectory_msgs/JointTrajectory` out.
  `type: unitree_arm` (`unitree_arm.py`) drives ONE arm of a standing
  G1 / H1 / H1-2 over the Arm-SDK channel (`rt/arm_sdk` LowCmd,
  `rt/lowstate`), with balance, legs and waist explicitly left to Unitree's
  own controller. Nothing subscribes to odometry, publishes a Twist, talks
  to Nav2, reads an `OccupancyGrid` or estimates the robot's own pose.
- **"Localization" in this repo means object grounding** (`localize_object`,
  `probe_point`, `locate_pixel`, `perception/grounding.py`): where is the
  cup in the base frame. Robot self-localization (where is the base in the
  map) does not exist.
- **The occupancy service is a 3D TSDF/ESDF for the ARM** (`perception/
  occupancy_backends.py`: nvblox | warp | voxel; `query(region)` returns
  distance-to-nearest-obstacle). It integrates depth from the rig cameras
  with the robot body masked out. It is exactly the data a 2D costmap for a
  base needs, at 1 cm resolution, already on a Warp device.
- **The memory harness is planner-agnostic** (`memory/episodic.py::
  memory_frames(k)`, injected by `orchestrator._with_memory_harness`,
  served by the `task_memory` tool): first frame pinned, uniform sample,
  newest last, captions = action + independent verdict.
- **The simulators**: MuJoCo via a shared world registry (`sim/mujoco_world.py`,
  arm + rendered cameras + truth channel on one `MjData`); Isaac Sim via a
  newline-JSON TCP bridge (`scripts/isaac_bridge.py`, ops `ping frame state
  set_joints gripper stop place_prop reset_props exec`).

## What Vesta does -- and does not -- do for navigation

arXiv:2606.20905 §2.2 (read in full; no weights or code released):

- Vesta is a **VLN planner**. At each high-level decision step it receives
  the route instruction, the current egocentric frame and up to N *sampled
  history frames* (the same harness as its manipulation memory), and emits
  ONE of three actions: a **pixel goal** (`↓` + normalized `(u, v) ∈
  [0,1000]²` on a downward view), a **turn sequence** over `←`/`→` yaw
  primitives, or **stop**.
- "Low-level motion is handled by the navigation backend" (Wei et al.
  2025b). Vesta does **no** mapping, SLAM, odometry or metric localization.
- Its "localization" (§2.1) is grounding -- points and boxes as text tokens
  -- which cascade already has.
- Evaluated in Habitat (R2R val-unseen, 1839 episodes); its real-robot
  results are bimanual tabletop manipulation, not navigation.

So the Vesta-shaped thing to build is: **a planner-facing navigation skill
API of three verbs, a memory harness that spans the walk, and a pluggable
backend that turns a pixel goal into motion**. Mapping and localization are
the backend's job, and there are two mature backends to plug.

## Interface: `MobileBase` (twin of `ArmBase`)

```python
class MobileBase(ABC):                         # control/base_base.py
    def connect(self) -> None
    def disconnect(self) -> None
    def get_pose(self) -> BasePose             # x, y, yaw in the MAP frame if localized, else ODOM; frame + covariance named
    def set_velocity(self, vx: float, vy: float, wz: float) -> None   # body frame; the backend re-sends at its own rate and zeroes on silence
    def go_to(self, x: float, y: float, yaw: float | None, frame: str) -> NavGoal   # non-blocking handle: status, progress, cancel()
    def stop(self) -> None                     # zero velocity AND cancel any goal; latches like the arm e-stop
    def costmap(self) -> Costmap | None        # 2D occupancy for the planner + the safety gate; None if the backend has none
```

Profiles in `configs/bases/*.yaml` select the backend by `type:` and carry
everything that varies per robot: footprint polygon, max `vx/vy/wz`,
acceleration caps, stand height, the camera(s) rigidly attached to the
body and their `T_base_cam`, the arm(s) mounted on it (`arms: [g1_left]`),
and the frame names (`map`, `odom`, `base_link`). One physical robot,
several transports, `extends:` -- the same rules as arms.

The **rig** grows one member: `MobileRig` next to `ArmRig` and `CameraRig`;
`build_runtime` wires it; `SkillRuntime.execute()` binds `base=` per call
exactly as it binds `arm=`. When a humanoid carries an arm, the arm's
`base_pose` (already a profile key, `types.pose_to_transform`) becomes the
base's live pose instead of a constant: the inter-arm gate and the
occupancy gate keep working because every position was already lifted into
a shared frame.

## Backends

| `type:` | robot | motion | pose | costmap | status |
|---|---|---|---|---|---|
| `mock_base` | any | integrates commands kinematically | odom = integrated | none | tests / CI |
| `mujoco_base` | Go2 / G1 / H1 from Menagerie, floating base in the shared world | velocity → base joint targets via a locomotion policy **or** a kinematic "hover" mode for arm-first work | `data.xpos` (truth) | from the shared world's geometry (truth) | sim, laptop |
| `isaac_base` | G1 / H1 with the Isaac Lab locomotion policy | bridge op `set_velocity` → policy | bridge op `state` + `base_pose` | Isaac RTX depth into the occupancy service | sim, GPU |
| `ros2_base` | ANY robot with `/cmd_vel` + `/odom` (+ Nav2) | `geometry_msgs/Twist` on `/cmd_vel`; `nav2_msgs/action/NavigateToPose` when Nav2 is up | `/odom` and `tf` `map→base_link` (AMCL / SLAM Toolbox) | `nav2_msgs/Costmap` or `/map` `OccupancyGrid` | the standard path |
| `unitree_base` | G1 / H1-2 over `unitree_sdk2` **LocoClient** (`Move(vx, vy, vyaw)`, `StopMove`, `Damp`, `StandUp`, `SetVelocity(vx,vy,ω,duration)`, FSM ids) | SDK RPC | sport-mode state (odom) | none (uses ours) | hardware, unverified |

**The two navigation backends behind `go_to`:**

1. **Nav2** (`ros2_base` with `nav2_msgs` importable): `NavigateToPose`
   action client; progress from feedback (`distance_remaining`), map and
   costmaps from Nav2, localization from AMCL or SLAM Toolbox. Heavy, standard,
   and what a real G1 deployment will have -- NVIDIA's Isaac ROS Deploy ships
   the G1 AGILE whole-body locomotion policy behind `ros2_control` driven by
   `/cmd_vel` Twist, with `hardware_type:=mujoco` for sim and the same
   interface on the real robot (`ros-jazzy-unitree-g1-bringup`). That is the
   reference target: **cascade publishes Twist and goals; the WBC policy
   walks.**
2. **Warp planner** (`perception/nav_planner.py`, no ROS2): a 2D costmap
   sliced from the existing Warp TSDF/ESDF (`WarpTsdfBackend`: a
   `z ∈ [0.05, 1.6] m` max-projection is the traversability layer; the ESDF
   IS the inflation), Theta*/A* on it, a pure-pursuit velocity follower
   feeding `set_velocity`. Pose from whatever the base reports (truth in
   sim). This keeps `./run.sh mujoco --base go2_mujoco` working on a laptop
   with no ROS2, and is what the *tests* run.

`MobileBase.go_to()` picks Nav2 when `rclpy` + `nav2_msgs` import and an
action server answers within the connect timeout; otherwise the Warp planner.
The banner says which one, like it does for grasp and occupancy sidecars.

## Skills (Vesta's three verbs, plus what a chat host needs)

Added to `TOOL_SPECS` like any skill; `_MOTION_SKILLS` gains them so belief
fusion pauses and a memory frame + verdict is recorded after each.

| skill | Vesta action | what it does |
|---|---|---|
| `go_to_pixel(u, v, camera)` | pixel goal | the pixel is lifted through depth to a FLOOR point in the map/odom frame (reject if not on the floor plane), handed to `go_to`; returns `NavGoal` id |
| `turn(direction, deg)` | turn sequence | yaw primitive; `←`/`→` in Vesta's vocabulary |
| `stop_navigation()` | stop | cancel + zero velocity |
| `go_to_object(label)` | -- | belief → floor point at standoff distance, facing the object (what a manipulation planner actually asks for) |
| `where_am_i()` | -- | pose, frame, localization quality, distance travelled this task |
| `look(direction)` / `camera_snapshot(camera="head")` | -- | Vesta's "downward-view request `↓`" needs a view the planner can point at |

Postconditions (`agent/effects.py`, new kind `nav`): `go_to*` is CONFIRMED
when the pose channel says the base is within `d_succ` of the goal,
REFUTED when it moved away or timed out, UNVERIFIED when no pose channel.
In MuJoCo/Isaac the pose channel is physics truth -- the same
`channel: physics` the arm skills report.

Safety (`safety/base_harness.py`): speed caps from the profile, a
geofence AABB, costmap clearance for the footprint at every velocity
tick, e-stop shared with the arm harness (one latch for the whole robot),
and **arm-in-motion → base frozen** (a humanoid must not walk while the
arm harness is streaming a grasp). Fail closed, like the arm's.

## The planner sees the walk (memory harness)

Nothing new to build: `memory_frames(k)` already samples the task-scale
frame ring with the first frame pinned. What changes is what a "step" is:
each `go_to*`/`turn` records its AFTER frame with the caption
`moved 1.8 m to (x, y); pose channel: physics; CONFIRMED`. The planner's
prompt gains the VLN decision form (instruction, current view, history,
one of: pixel goal / turn / stop / manipulation skill) -- Vesta's finding
that images + text beats text alone is the reason this is worth doing for
navigation too, and their Table 4 (unified model beats a nav-only
specialist on nav *and* embodied) is the reason it is one planner, not a
navigation specialist beside the manipulation one.

## Demo task where mobility is visible (the Vesta rule)

A two-room task: *"the red cube is on one of the two tables; find it, bring
it to the blue tray."* After turning away from table A, the current view
alone cannot say whether A was already checked -- only the history frames
can. Same logic as the two-cube task, one level up. Postconditions: `go_to`
confirmed by pose truth, `pick_and_place` confirmed by prop truth; the
judge scores the walk frames like it scores the pick frames.

## Order of work (each step has a test that can fail and a real run)

1. `MobileBase` + `mock_base` + `MobileRig` + `base=` binding in
   `execute()`; profile loader for `configs/bases/`; tests mirror
   `tests/test_arm_rig.py`.
2. `mujoco_base` on the shared world with a **floating-base kinematic mode**
   (the base pose is set, not walked -- honest about it in the banner) and
   Menagerie Go2/G1; pose truth channel; `where_am_i`, `turn`,
   `go_to_pixel` on the Warp planner; the two-room scene; suite count moves,
   0 skipped.
3. `go_to` postconditions + base harness (geofence, footprint clearance,
   arm-in-motion freeze) with mutation-checked tests.
4. `isaac_base`: extend `isaac_bridge.py` with `set_velocity` / `base_pose`
   ops and load a G1 with its Isaac Lab velocity policy; the one-click
   `--sim isaac --base g1_isaac` proves a walk + a pick.
5. `ros2_base` (+ Nav2 client) against Isaac ROS Deploy's G1 bringup in
   `hardware_type:=mujoco`: the real-robot interface, no robot.
6. `unitree_base` over `LocoClient` -- written from the SDK headers,
   marked UNVERIFIED like `unitree_arm` until a robot is in the room.

Not adopted, with reasons: Vesta as the brain (no weights); training a VLN
policy (we plan, we do not learn to plan); SLAM of our own (Nav2's stack or
sim truth; a third mapper would be a distraction from the planner, which is
the part that is ours).
