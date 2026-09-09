# Design brief: `type: ros2` backend for cascade — findings from ros-claw/rosclaw

Source: https://github.com/ros-claw/rosclaw @ `6620b1b` (2026-09-03), v1.2.0, alpha.
Read from a shallow clone; all `file:line` refs are against that commit. Note the
namespace collision: `PlaiPin/rosclaw` (TypeScript OpenClaw plugin, Apache-2.0) and
`tranzmatt/rosclaw.ros-claw` are unrelated projects — everything below is the
canonical `ros-claw/rosclaw` (MIT).

## ARCHITECTURE

ROSClaw is an MCP-fronted "execution runtime": agents (Claude Code/Codex/OpenClaw) are
northbound MCP clients; ROS 2 / vendor SDKs / MuJoCo are southbound (README.md:30-41).
Canonical path: intent → body/capability binding → `rosclawd` permit → resource lease →
dispatch → physical observation → `ExecutionReceipt` (README.md:106-111). It has **two**
ROS transports:

1. **Primary: rosbridge WebSocket, deliberately no rclpy.** `RosTransport` Protocol
   (`connectors/ros/transport/base.py:57-71`) states "Implementations must not import ROS
   Python client libraries"; `transport/rosbridge.py` implements `publish` (:280),
   `subscribe_once` (:287), `call_service` (:244), `advertise` (:260) as rosbridge JSON ops.
   Rationale (docs/ROS_CONNECTOR.md:3-6): installable on machines without ROS. Actions ride
   the same socket: `connectors/ros/action_client.py:46-133` implements
   `send_goal`/`cancel_goal` + `action_feedback`/`action_result` push on a listener thread
   (feedback only arrives on the goal-origin connection — single-transport constraint,
   action_client.py:32-36).
2. **Secondary: native rclpy driver** (`mcp_drivers/ros2_driver.py`). Topics only:
   - publish `trajectory_msgs/JointTrajectory` → `/joint_trajectory_controller/joint_trajectory`, depth 10 (:50-52)
   - subscribe `sensor_msgs/JointState` ← `/joint_states`, depth 10 (:53-55), cached to a dict (:76-91)
   - rclpy import failure → degrade to "fixture mode", never crash (:63-66).

Message types actually used anywhere in the repo: `sensor_msgs/JointState`,
`trajectory_msgs/JointTrajectory(+Point)`, `geometry_msgs/Pose` (ur5_server.py:66-67),
`geometry_msgs/Twist` (`/cmd_vel`, guarded-base path), `moveit_msgs/action/MoveGroup` and
`std_srvs/srv/Trigger` (declared in the ur5e embodiment card,
`connectors/ros/specs/ur5e_moveit.yaml:9-25`). **`control_msgs`/`FollowJointTrajectory` is
never used** (verified grep across src/docs/configs). `sensor_msgs/Image` appears only as a
discovery pattern (`*/image_raw`, discovery/graph.py:35), never as a subscribed type.

Topics-vs-services-vs-actions policy is data, not code: per-robot "embodiment cards"
(specs/*.yaml) rank `preferred_interfaces` (UR5e: MoveIt `/move_action` action, "plans and
executes collision-checked trajectories"; gripper via Trigger services) and
`discouraged_interfaces` (raw `/joint_command` topic, "bypasses MoveIt collision checking",
ur5e_moveit.yaml:27-30).

### Capability discovery / exposure to the agent

- Live discovery through rosbridge `/rosapi/*` services → `RosGraphSnapshot`
  (discovery/graph.py); topics classified by glob heuristics: `COMMAND_TOPIC_PATTERNS`
  (`/cmd_vel`, `*/joint_trajectory`, `*/effort_controller/*`, graph.py:24-32) vs
  `SENSOR_TOPIC_PATTERNS` (`*/joint_states`, `*/image_raw`, `*/scan`, `*/odom`, :34-43).
- A compiler turns the snapshot into a `CapabilityManifest` of `RosCapability` records —
  `{id, kind: observation|actuation|state|parameter, interface{ros_kind,name,msg_type},
  schema, risk{level, read_only, destructive, requires_stop_guard, max_duration_sec,
  max_rate_hz}}` (compiler/capability_manifest.py:22-60).
- `SafetyContractCompiler` compiles risk metadata into per-capability rules and evaluates
  each call → `ALLOW / MODIFY / BLOCK` (compiler/safety_contract.py:158-227).
- MCP exposes **only compiled capabilities** — "raw `publish_once` / `call_any_service` are
  intentionally not exposed" (docs/ROS_CONNECTOR.md:113-115).
- `body/ros_introspection.py:49-95` folds the snapshot into a conservative `body.yaml`
  runtime-state patch (active joint_state/camera/lidar/odom topics, command topics).

### Install / launch

Plain **pip package** (pyproject.toml, setuptools, console scripts `rosclaw`, `rosclawd`,
`rosclaw-ur5-mcp`). **No colcon**: zero `package.xml`, zero ament references. The `ros2`
extra is an empty marker — "ROS 2 Python/message packages are supplied by the selected ROS
distribution (apt/rosdep), not PyPI" (pyproject.toml optional-dependencies comment).
Robot-side launch is stock `ros2 launch rosbridge_server rosbridge_websocket_launch.xml`;
client default endpoint `ws://127.0.0.1:9090` with `--endpoint` override. Docker-compose
turtlesim stacks back the integration tests, which skip when no server is reachable
(docs/ROS_CONNECTOR.md:37-61; env `ROSCLAW_ROS_TEST_ENDPOINT`).

### Safety / e-stop

- Fail-closed doctrine (docs/SAFETY.md:121-142): sandbox indecision → BLOCK; daemon down →
  all physical requests fail; interrupted REAL action is never auto-retried (daemon latches
  E-Stop, records unknown outcome, blocks new REAL work pending human review); "a software
  E-stop dispatch or driver ACK is not reported as a successful physical stop without
  physical stop observation" (:141-142).
- `rosclawd` is a separate-UID daemon owning the E-Stop latch, permits, leases, watchdogs,
  receipts; agent processes cannot self-approve REAL actions (README.md:118-131); deploy
  guidance: agent user gets no device groups and no ROS 2 command-topic permissions, SROS2
  for DDS access control (SAFETY.md:107-110).
- Contract-level guards: velocity commands **require a bounded duration** (missing →
  violation; over-limit → clamped, safety_contract.py:188-195); numeric args clamped into
  per-key bounds (:198-205); HIGH_RISK violations BLOCK instead of MODIFY (:208-214);
  forbidden name patterns (`torque_control`, `disable_safety`, `motor_reset`, …) are
  unconditionally blocked (:104-112,127-129); `/cmd_vel` is hardcoded HIGH_RISK (:147).
- Deadman, proven in sim: guarded Gazebo base must emit `deadman_stop`
  (`reason: fresh_command_timeout`) within ≤1.25 s of killing rosbridge
  (simforge/gazebo_guarded_base.py:551-573).
- E-stop *implementation* is honest but weak: `ros2_driver.emergency_stop()` publishes a
  hold-position single-point JointTrajectory (current positions, `time_from_start` = 0.1 s)
  and returns `acknowledged: False, physical_stop_observed: False` with an explanatory note
  (ros2_driver.py:157-186). The UR5 MCP node is observation-only; its motion/e-stop
  surfaces hard-fail with "use rosclawd request_action" (mcp/ur5_server.py:211-229).
- Pre-dispatch MuJoCo "Digital Twin firewall" forward-simulates commands against joint and
  torque limits (safety_margin 0.05) when a model is available (ur5_server.py:262-273).

## MESSAGE FLOW

The only end-to-end **arm** flow in the repo (rclpy path, `mcp_drivers/ros2_driver.py`):

```
agent tool call → BaseDriver.move_joints(positions, duration)      ros2_driver.py:108
  → _validate_joint_positions/_validate_duration                    :110-111
  → build JointTrajectory{joint_names=[f"joint_{i}"...],            :119-126
      points=[JointTrajectoryPoint(positions,
              time_from_start.sec=int(duration))]}    ← BUG: sub-second truncates to 0
  → publisher.publish → /joint_trajectory_controller/joint_trajectory  :127
feedback: /joint_states → _on_joint_state → cached dict            :53-55, 76-91
  → get_joint_positions() reads cache (zeros until first msg)      :93-96
e-stop: publish hold-position JointTrajectory, report unverified   :157-186
```

The rosbridge flow (turtlesim/Gazebo, the only *verified* one) is
`MCP tool → provider._execute_capability → SafetyContract.evaluate → rosclawd permit →
rosbridge {"op":"publish","topic":"/cmd_vel",...}` — and note the legacy provider now
**refuses all non-dry-run execution** pending ActionGateway registration
(provider/ros_capability_provider.py:346-377). QoS on the one real rclpy subscriber:
`QoSProfile(depth=10, reliability=BEST_EFFORT)` for `/{namespace}/joint_states`
(ur5_server.py:153-158), with joint-name reorder mapping via `msg.name.index(name)`
(:174-183). Import-time rclpy stubbing for no-ROS machines: ur5_server.py:19-43.

## WHAT TO MIRROR (mapped to cascade's ArmBase / make_arm / stream_to)

1. **Lazy import + graceful degrade.** rosclaw does it three ways (driver try/except →
   fixture mode ros2_driver.py:35-66; import-time stub classes ur5_server.py:26-43; lazy
   `_ensure_rclpy` sense/collectors/ros2_collector.py:32-42). For cascade: keep every
   `import rclpy` inside `Ros2Arm.connect()` (or module import guarded), so `make_arm()`
   lazy import passes CI, and the planned `sys.modules["rclpy"]` stub slots in cleanly.
2. **JointState → cached state dict.** Subscriber callback caches
   positions/velocities/efforts; `get_state()` reads the cache and reports staleness
   (ros2_driver.py:76-106). Maps 1:1 onto `ArmBase.get_state()`. Adopt their "readiness =
   first feedback received" evidence gate (`_mark_backend_ready` on first `/joint_states`
   msg, :85-91): `connect()` should block (with timeout) until one JointState arrives, else
   fail — never return zeros as if they were a pose.
3. **Joint-name mapping, never index order.** ur5_server.py:174-183 reorders by
   `msg.name.index(name)`. Cascade profiles must carry `joint_names:` (controller-exact)
   and map both directions; rosclaw's own driver violates this with `f"joint_{i}"`
   placeholders (see AVOID #3).
4. **QoS split.** BEST_EFFORT/KEEP_LAST(10) for the `/joint_states` subscription
   (ur5_server.py:153) — compatible with both RELIABLE (ros2_control
   joint_state_broadcaster default) and BEST_EFFORT publishers, per ROS 2 QoS
   compatibility rules (docs.ros.org Jazzy: Concepts → Quality of Service, "best effort"
   sub matches any pub reliability). RELIABLE (rclpy default) KEEP_LAST(1-10) for the
   command publisher. Use `rclpy.qos.qos_profile_sensor_data` for the subscriber.
5. **Per-profile namespacing.** `f"/{namespace}/joint_states"` (ur5_server.py:136,158).
   Cascade YAML: `namespace:`, `joint_states_topic:`, `command_topic:` (and optional
   `controller_action:`) per profile, defaults derived from namespace — this is exactly the
   per-profile topic-name plan, and it is what lets two `ros2` arms coexist in one ArmRig.
6. **rclpy context discipline.** `if not rclpy.ok(): rclpy.init()` on connect
   (ros2_driver.py:46-47) and **never `rclpy.shutdown()` in disconnect** — "it's global
   state; multiple test instances share the same rclpy context" (:72-73). Destroy the node
   only. Critical for cascade: a two-arm ArmRig means two Ros2Arm instances in one process
   — one shared context, one spin thread (SingleThreadedExecutor with both nodes, or one
   node per arm each spun via `rclpy.spin(node)` in a daemon thread).
7. **E-stop evidence honesty.** Return structured `{acknowledged, physical_stop_observed}`
   rather than claiming success (ros2_driver.py:157-186; SAFETY.md:141-142).
   `Ros2Arm.stop()`: cancel any active goal / publish hold-position trajectory, then report
   what was observed; the SafetyHarness e-stop latch stays the authoritative gate.
8. **Stop-guard/deadman as contract.** rosclaw refuses unbounded velocity commands and
   proves bounded stop on link loss (safety_contract.py:188-195;
   gazebo_guarded_base.py:551-573). Cascade's 50 Hz `stream_to()` is a natural deadman
   *only if* the receiving controller has a command timeout — record that requirement in
   the profile (`requires_controller_watchdog: true` doc note) and prefer
   FollowJointTrajectory (cancel = abort) for point-to-point moves.
9. **Curated agent surface.** Raw publish/service tools are deliberately not exposed to
   the agent (ROS_CONNECTOR.md:113-115). Cascade already has this property (skills over
   SafeArm) — keep it; do not add a generic `ros2_publish` skill.
10. **Preferred/discouraged interface cards.** The specs/*.yaml embodiment-card shape
    (preferred vs discouraged interfaces + preconditions like `e_stop_not_active`,
    ur5e_moveit.yaml:27-43) is a good pattern for cascade's profile YAML if a profile ever
    exposes both a streaming topic and a MoveIt action.

## WHAT TO AVOID

1. **rosbridge as the arm transport.** rosclaw's core pays WebSocket+JSON per message and
   requires rosbridge_server on the robot. Fine for 1 Hz `cmd_vel` bursts; wrong for
   cascade's 50 Hz interpolated stream with per-waypoint harness approval. Use native
   rclpy/DDS (their own arm driver does).
2. **`time_from_start.sec = int(duration)`** (ros2_driver.py:125,148) truncates sub-second
   durations to 0 — a live bug at 50 Hz waypoint spacing (0.02 s → 0). Always set
   `sec` + `nanosec` (or `rclpy.duration.Duration(seconds=...).to_msg()`).
3. **Placeholder joint names** `f"joint_{i}"` (ros2_driver.py:122,144) — real
   `ros2_control` trajectory controllers reject unknown names. Profile-supplied names only.
4. **One-point JointTrajectory per waypoint.** Republishing single-point trajectories at
   50 Hz makes the trajectory controller restart interpolation every tick. For cascade's
   `stream_to()`: either (a) target a streaming controller
   (`position_controllers/JointGroupPositionController`, `std_msgs/Float64MultiArray`) for
   the 50 Hz path, or (b) send `FollowJointTrajectory` goals with the whole interpolated
   segment and abort via action cancel when the harness trips — do **not** copy rosclaw's
   republish pattern. (rosclaw never uses `control_msgs` at all; there is no action-based
   arm motion in the repo to copy.)
5. **The daemon/permit/ledger stack** (rosclawd, HMAC ledger, leases, receipts). Right for
   their threat model (untrusted agent processes on shared hosts); overkill for cascade
   where SafetyHarness gates every waypoint in-process and `SafeArm` is the boundary.
6. **Assuming the code is hardware-proven.** The legacy provider blocks all non-dry-run
   execution (ros_capability_provider.py:346-377); README's own status table scopes
   verification to MuJoCo sim + turtlesim/Gazebo `cmd_vel` (README.md:49-64). Treat rosclaw
   as an architecture reference, not as validated arm-control code.
7. **Hardcoded global topic defaults** (`/joint_trajectory_controller/joint_trajectory`,
   ros2_driver.py:51) without per-robot override — contradicts multi-arm rigs.
8. **Name confusion.** Cite `ros-claw/rosclaw@6620b1b` explicitly; `PlaiPin/rosclaw`
   (Apache-2.0, TypeScript, rosbridge-only) is a different project with the same name.

## LICENSE

**MIT** — `LICENSE:1-3` ("MIT License / Copyright (c) 2026 ROSClaw Team") and
`pyproject.toml` `license = {text = "MIT"}`. Permissive; compatible with reading, porting
patterns, or vendoring snippets into cascade with attribution. `THIRD_PARTY.md` /
`THIRD_PARTY_NOTICES.md` exist for their bundled deps. (Older third-party listings claiming
Apache-2.0 refer to forks or the unrelated PlaiPin project.)

## Concrete sketch for cascade `type: ros2`

```yaml
# configs/arms/ur5e_ros2.yaml
type: ros2
namespace: ur            # → /ur/joint_states etc. unless overridden
joint_names: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint,
              wrist_1_joint, wrist_2_joint, wrist_3_joint]
joint_states_topic: /ur/joint_states          # sub, qos_profile_sensor_data
command_topic: /ur/forward_position_controller/commands   # 50 Hz stream path
controller_action: /ur/joint_trajectory_controller/follow_joint_trajectory  # optional, point-to-point
state_timeout_s: 1.0     # connect() fails if no JointState within this window
```

`Ros2Arm(ArmBase)`: `connect()` = lazy `import rclpy` → `rclpy.init()` if not ok → node
`cascade_arm_<name>` → sub + pub(s) → spin thread → wait for first JointState;
`get_state()` = cached JointState remapped by name (stale ⇒ raise); `send_joint_target()` =
one Float64MultiArray command publish (called by `stream_to()` at 50 Hz after
`harness.approve()`); `stop()` = cancel active FollowJointTrajectory goal + publish
hold-position + report `physical_stop_observed` from subsequent velocity feedback;
`disconnect()` = destroy node, never `rclpy.shutdown()`. CI: inject stub `rclpy`,
`rclpy.node`, `rclpy.qos`, `sensor_msgs.msg`, `trajectory_msgs.msg`, `builtin_interfaces.msg`
into `sys.modules` before `make_arm()` — mirror the ur5_server.py:26-43 stub shapes.
