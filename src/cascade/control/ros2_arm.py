"""Any ROS2 robot as a cascade arm (`type: ros2`).

The transport speaks the two interfaces every ros2_control deployment (and
the rosclaw bridge) already exposes, so "adding a robot" is a profile file
naming its topics -- no Python:

  state    sensor_msgs/JointState   subscribed with SENSOR_DATA QoS
           (best-effort matches both best-effort and reliable publishers;
           a RELIABLE subscription to a best-effort driver never pairs in
           DDS and reads as "the arm publishes nothing").
  command  control_msgs/FollowJointTrajectory, one smooth, monitored motion
           when `ros_trajectory_action` is configured; otherwise
           trajectory_msgs/JointTrajectory, one point per streamed waypoint
           (`ros_command: trajectory`, the joint_trajectory_controller
           path), or std_msgs/Float64MultiArray for a
           forward_position_controller (`ros_command: position`).
  gripper  same publisher family on `ros_gripper_topic` (default), or a
           control_msgs/GripperCommand ACTION (`ros_gripper: action`,
           `ros_gripper_action: <name>`) for drivers like franka_gripper
           and robotiq that expose no trajectory topic for the fingers.
           `ros_gripper_state_topic` adds a second JointState subscription
           when the fingers are published separately from the arm.

           A single position-only JointTrajectory per waypoint makes a JTC
           restart linear interpolation every 20 ms and request zero
           velocity whenever a point expires. Effort JTC profiles should
           configure `ros_trajectory_action` for complete quintic motions.
           For a robot
           whose bringup offers a forward/JointGroupPositionController,
           prefer `ros_command: position` for the streaming path; keep
           `trajectory` (the default) for stock bringups that ship only a
           JTC. Point-to-point moves are fine on either.

Joints are addressed BY NAME, never by index: a JointState's order is
whatever the driver publishes (arm + gripper interleaved, alphabetical,
split across messages), and any index would silently shift. The profile's
`ros_joints` list defines this framework's q order; each incoming message
updates only the names it carries.

`stop()` in trajectory mode publishes a named, measured hold with zero velocity
and suppresses further waypoints until `resume()`, mirroring the Feetech
soft stop (torque stays on, nothing falls). Position mode re-asserts the
last commanded target instead, because a forward controller has no cancel.

Heavy imports (rclpy, message packages) stay inside connect(), so the mock
stack imports nothing and CI without ROS2 still collects this module; the
unit tests inject a stub rclpy into sys.modules (same convention as the
stubbed openai client in test_hermes_brain.py).
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from ..config import Cfg
from ..types import GraspFailed, MotionHalted, RobotState
from .arm_base import ArmBase, PREFLIGHT_MAX_DRIFT_RAD, min_jerk

logger = logging.getLogger(__name__)


class Ros2Arm(ArmBase):
    # ros2_control retains the drives after this client disconnects. Shutdown
    # must not park or close the hand; disconnect only cancels an active goal
    # into a measured hold before releasing the transport.
    shutdown_park_required = False

    def __init__(self, cfg: Cfg):
        self._cfg = cfg
        self.n_joints = int(cfg.get("n_joints", 6))
        self.settle_tol = float(cfg.get("settle_tol", 0.03))
        self.settle_timeout_s = float(cfg.get("settle_timeout_s", 3.0))
        self._trajectory_velocity_limit = self._joint_limit_vector("ros_max_joint_velocity_rad_s")
        self._trajectory_acceleration_limit = self._joint_limit_vector("ros_max_joint_acceleration_rad_s2")

        self._joints = [str(j) for j in (cfg.get("ros_joints") or [])]
        if len(self._joints) != self.n_joints:
            raise ValueError(
                f"ros_joints has {len(self._joints)} entries but "
                f"n_joints={self.n_joints}; the q order must be declared "
                f"explicitly (JointState order is driver-defined)"
            )
        self._ns = str(cfg.get("ros_namespace", "")).strip().strip("/")
        self._mode = str(cfg.get("ros_command", "trajectory")).lower()
        if self._mode not in ("trajectory", "position"):
            raise ValueError(f"ros_command must be trajectory|position, got {self._mode!r}")
        self._state_topic = self._resolve(str(cfg.get("ros_state_topic", "joint_states")))
        default_cmd = ("joint_trajectory_controller/joint_trajectory"
                       if self._mode == "trajectory"
                       else "forward_position_controller/commands")
        self._cmd_topic = self._resolve(str(cfg.get("ros_command_topic", default_cmd)))
        trajectory_action = cfg.get("ros_trajectory_action")
        self._trajectory_action = self._resolve(str(trajectory_action)) if trajectory_action else None
        if self._trajectory_action and self._mode != "trajectory":
            raise ValueError("ros_trajectory_action requires ros_command: trajectory")
        self._trajectory_client = None
        self._arm_goal = None
        self._motion_generation = 0
        # Dispatch and stop share a lock: a stop cannot slip between the
        # final generation check and submission of a queued trajectory.
        self._command_lock = threading.RLock()
        self._motion_lock = threading.Lock()
        self._motion_active = False
        self._motion_error = None
        self._robot_state_topic = cfg.get("ros_robot_state_topic")
        self._robot_monitor = None
        self._grip_joint = cfg.get("ros_gripper_joint")
        self._grip_joint = str(self._grip_joint) if self._grip_joint else None
        gt = cfg.get("ros_gripper_topic")
        self._grip_topic = self._resolve(str(gt)) if gt else None
        # Gripper transport. `topic` (default): a JointTrajectory /
        # Float64MultiArray publisher, same as the arm. `action`: a
        # control_msgs/GripperCommand action server -- what franka_gripper,
        # robotiq's driver and MoveIt's gripper controllers expose. The action
        # goal carries `position` = the finger JOINT value in the profile's
        # gripper units (Franka: one finger 0..0.04 m, i.e. half the width)
        # and `max_effort` (N), and the server itself holds the goal, so
        # "close on an object" is a stalled-goal result, not a timeout.
        self._grip_mode = str(cfg.get("ros_gripper", "topic")).lower()
        if self._grip_mode not in ("topic", "action"):
            raise ValueError(f"ros_gripper must be topic|action, got {self._grip_mode!r}")
        ga = cfg.get("ros_gripper_action")
        self._grip_action = self._resolve(str(ga)) if ga else None
        if self._grip_mode == "action" and not (self._grip_action and self._grip_joint):
            raise ValueError("ros_gripper: action needs ros_gripper_action and ros_gripper_joint")
        # Some drivers (franka_gripper) publish the finger joints on their OWN
        # JointState topic instead of the arm's; subscribe to both when set.
        gst = cfg.get("ros_gripper_state_topic")
        self._grip_state_topic = self._resolve(str(gst)) if gst else None
        self._grip_max_effort = float((cfg.get("gripper") or {}).get("max_effort_n", 20.0))
        self._grip_min_effort = float((cfg.get("gripper") or {}).get("min_effort_n", 0.0))
        self._grip_goal = None
        self._grip_error = None
        self._grip_generation = 0
        grasp_action = cfg.get("ros_gripper_grasp_action")
        self._grasp_action = self._resolve(str(grasp_action)) if grasp_action else None
        self._grasp_client = None
        move_action = cfg.get("ros_gripper_move_action")
        self._grip_move_action = self._resolve(str(move_action)) if move_action else None
        self._grip_move_client = None
        self._grip_wait_s = float(cfg.get("ros_gripper_wait_s", 0.0))
        #: seconds a JointState may age before get_state() calls the arm lost
        self._state_timeout = float(cfg.get("state_timeout_s", 1.0))
        #: how long connect() waits for the FIRST JointState
        self._first_state_timeout = float(cfg.get("wait_state_s", 5.0))
        #: time_from_start for each streamed point; 20 ms matches the 50 Hz
        #: stream so the controller's interpolator never starves or lags.
        self._point_dt = float(cfg.get("ros_point_dt_s", 0.02))

        self._node = None
        self._executor = None
        self._spin_thread: threading.Thread | None = None
        self._cmd_pub = None
        self._grip_pub = None
        self._grip_client = None
        self.connected = False
        self._stopped = False
        self._last_cmd_q: np.ndarray | None = None
        self._lock = threading.Lock()
        # name -> latest position; filled by every JointState callback
        self._pos: dict[str, float] = {}
        self._joint_times: dict[str, float] = {}
        self._last_state_t: float | None = None
        self._gripper_pos = 0.0
        self._gripper_valid = False
        self._last_gripper_t = None

    def _resolve(self, topic: str) -> str:
        """Namespace-prefix relative topics; absolute (`/x`) pass through."""
        if topic.startswith("/"):
            return topic
        return f"/{self._ns}/{topic}" if self._ns else f"/{topic}"

    # ── lifecycle ────────────────────────────────────────────────────────

    def connect(self) -> None:
        if self.connected:
            return
        with self._lock:
            self._pos.clear()
            self._joint_times.clear()
            self._last_state_t = None
            self._last_gripper_t = None
            self._gripper_valid = False
            self._grip_error = None
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import JointState
        except ImportError as e:
            raise RuntimeError(
                "the ros2 backend needs rclpy + common_interfaces in this "
                "environment (source your ROS2 setup.bash, or run inside a "
                "ROS2 distro container)"
            ) from e

        if not rclpy.ok():
            # NEVER paired with rclpy.shutdown(): the context is global to
            # the process, and a dual-arm ArmRig runs two Ros2Arm instances
            # in it -- the first arm to disconnect would tear the context
            # down under the second one's spin thread. Init once, leave it.
            rclpy.init()
        name = str(self._cfg.get("name", "arm")).replace("-", "_")
        self._node = Node(f"cascade_{name}")

        self._node.create_subscription(
            JointState, self._state_topic, self._on_joint_state,
            qos_profile_sensor_data,
        )
        if self._grip_state_topic and self._grip_state_topic != self._state_topic:
            self._node.create_subscription(
                JointState, self._grip_state_topic, self._on_joint_state,
                qos_profile_sensor_data,
            )
        if self._robot_state_topic:
            from franka_msgs.msg import FrankaRobotState
            from .franka_state import FrankaStateMonitor

            self._robot_monitor = FrankaStateMonitor(timeout_s=self._state_timeout)
            self._node.create_subscription(
                FrankaRobotState, self._resolve(str(self._robot_state_topic)),
                self._robot_monitor.update, qos_profile_sensor_data,
            )
        if self._mode == "trajectory":
            from trajectory_msgs.msg import JointTrajectory

            self._cmd_pub = self._node.create_publisher(
                JointTrajectory, self._cmd_topic, 10)
            if self._grip_mode == "topic" and self._grip_topic and self._grip_joint:
                self._grip_pub = self._node.create_publisher(
                    JointTrajectory, self._grip_topic, 10)
        else:
            from std_msgs.msg import Float64MultiArray

            self._cmd_pub = self._node.create_publisher(
                Float64MultiArray, self._cmd_topic, 10)
            if self._grip_mode == "topic" and self._grip_topic and self._grip_joint:
                self._grip_pub = self._node.create_publisher(
                    Float64MultiArray, self._grip_topic, 10)
        if self._trajectory_action:
            from control_msgs.action import FollowJointTrajectory
            from rclpy.action import ActionClient

            self._trajectory_client = ActionClient(
                self._node, FollowJointTrajectory, self._trajectory_action)
        if self._grip_mode == "action":
            try:
                from control_msgs.action import GripperCommand
                from rclpy.action import ActionClient
            except ImportError as e:
                raise RuntimeError(
                    "ros_gripper: action needs control_msgs (ros-<distro>-control-msgs)"
                ) from e
            self._grip_client = ActionClient(self._node, GripperCommand, self._grip_action)
            if self._grasp_action:
                from franka_msgs.action import Grasp

                self._grasp_client = ActionClient(self._node, Grasp, self._grasp_action)
            if self._grip_move_action:
                from franka_msgs.action import Move

                self._grip_move_client = ActionClient(self._node, Move, self._grip_move_action)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, name=f"ros2-spin-{name}", daemon=True)
        self._spin_thread.start()

        # Fail loudly if nobody publishes: a wrong namespace/topic is THE
        # classic ROS2 wiring error and it must not surface later as a
        # safety-harness staleness abort mid-motion.
        deadline = time.monotonic() + self._first_state_timeout
        while time.monotonic() < deadline:
            with self._lock:
                have = all(j in self._pos for j in self._joints)
            if have:
                break
            time.sleep(0.02)
        else:
            self.disconnect()
            raise RuntimeError(
                f"no JointState covering {self._joints} on {self._state_topic} "
                f"within {self._first_state_timeout:.0f}s -- check the "
                f"namespace, the controller is spawned, and QoS compatibility"
            )
        self.connected = True
        self._stopped = False
        state = self.get_state()
        self._last_cmd_q = state.q.copy()
        logger.info("ros2 arm up: state=%s cmd=%s (%s) q=%s",
                    self._state_topic, self._cmd_topic, self._mode,
                    np.round(state.q, 3).tolist())

    def disconnect(self) -> None:
        """Stop spinning and destroy the node. The CONTROLLER keeps the arm
        held (ros2_control owns the hardware) -- unlike serial backends there
        is no torque-off here, so nothing falls. rclpy's global context is
        deliberately NOT shut down (see connect)."""
        if self._motion_active or self._arm_goal is not None:
            self.stop()
        if self._executor is not None:
            try:
                self._executor.shutdown(timeout_sec=1.0)
            except Exception:  # noqa: BLE001
                pass
        if self._spin_thread is not None and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
        self._node = None
        self._executor = None
        self._spin_thread = None
        self._cmd_pub = None
        self._grip_pub = None
        self._grip_client = None
        self._grasp_client = None
        self._grip_move_client = None
        self._trajectory_client = None
        self._robot_monitor = None
        self.connected = False

    # ── feedback ─────────────────────────────────────────────────────────

    def _on_joint_state(self, msg) -> None:
        with self._lock:
            now = time.monotonic()
            for name, pos in zip(msg.name, msg.position):
                if not np.isfinite(pos):
                    continue
                self._pos[str(name)] = float(pos)
                self._joint_times[str(name)] = now
            if all(j in self._joint_times for j in self._joints):
                self._last_state_t = min(self._joint_times[j] for j in self._joints)
            if self._grip_joint and self._joint_times.get(self._grip_joint) == now:
                self._gripper_pos = self._pos[self._grip_joint]
                self._gripper_valid = True
                self._last_gripper_t = now

    def get_state(self) -> RobotState:
        return self._feedback_state(check_gripper_error=True)

    def _feedback_state(self, *, check_gripper_error: bool) -> RobotState:
        if not self.connected:
            raise RuntimeError("ros2 arm not connected")
        with self._lock:
            last_t = self._last_state_t
            if last_t is None or (time.monotonic() - last_t) > self._state_timeout:
                age = "never" if last_t is None else f"{time.monotonic() - last_t:.1f}s"
                raise RuntimeError(
                    f"JointState on {self._state_topic} is stale ({age}); "
                    f"driver/controller down?"
                )
            q = np.array([self._pos[j] for j in self._joints], dtype=float)
            if check_gripper_error and self._grip_error:
                raise RuntimeError(self._grip_error)
            gripper_valid = (self._gripper_valid and self._last_gripper_t is not None
                             and time.monotonic() - self._last_gripper_t <= self._state_timeout)
            return RobotState(
                q=q, gripper_pos=self._gripper_pos,
                gripper_valid=gripper_valid, t=last_t,
            )

    # ── commands ─────────────────────────────────────────────────────────

    def _traj_msg(self, joints: list[str], positions: list[float]):
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        msg = JointTrajectory()
        msg.joint_names = list(joints)
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in positions]
        sec = int(self._point_dt)
        pt.time_from_start = Duration(
            sec=sec, nanosec=int((self._point_dt - sec) * 1e9))
        msg.points = [pt]
        return msg

    def send_joint_target(self, q: np.ndarray) -> None:
        if not self.connected or self._cmd_pub is None:
            raise RuntimeError("ros2 arm not connected")
        q = np.asarray(q, dtype=float).reshape(-1)
        if len(q) != self.n_joints or not np.isfinite(q).all():
            raise ValueError(f"Expected {self.n_joints} finite joint targets")
        with self._command_lock:
            if self._stopped:
                return
            if self._mode == "trajectory":
                self._cmd_pub.publish(self._traj_msg(self._joints, q.tolist()))
            else:
                from std_msgs.msg import Float64MultiArray

                msg = Float64MultiArray()
                msg.data = q.tolist()
                self._cmd_pub.publish(msg)
            self._last_cmd_q = q.copy()

    @staticmethod
    def _duration(seconds: float):
        from builtin_interfaces.msg import Duration

        ns = int(round(seconds * 1e9))
        return Duration(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)

    @property
    def motion_status(self) -> dict:
        """Read readiness and the last motion failure without raising."""
        with self._command_lock:
            status = {"connected": self.connected, "stopped": self._stopped,
                      "motion_active": self._motion_active,
                      "motion_error": self._motion_error,
                      "trajectory_action": self._trajectory_action}
        with self._lock:
            age = None if self._last_state_t is None else max(0.0, time.monotonic() - self._last_state_t)
            feedback_valid = (age is not None and age <= self._state_timeout
                              and all(j in self._pos and np.isfinite(self._pos[j]) for j in self._joints))
        status.update(feedback_age_s=age, feedback_valid=feedback_valid)
        if self._robot_monitor is not None:
            status.update(self._robot_monitor.snapshot())
            status["motion_ready"] = (status["motion_ready"] and self.connected
                                      and not status["stopped"] and feedback_valid)
        else:
            status["motion_ready"] = self.connected and not status["stopped"] and feedback_valid
        return status

    def ensure_motion_ready(self) -> None:
        """Gate Franka skills without blocking read-only joint feedback."""
        if self._robot_monitor is not None:
            if self._stopped:
                raise RuntimeError("Arm is stopped; resume before starting another motion")
            self._robot_monitor.ensure_ready()
            self._feedback_state(check_gripper_error=False)

    def _ensure_motion_ready(self, generation: int):
        with self._command_lock:
            if self._stopped or generation != self._motion_generation:
                raise RuntimeError("Arm motion interrupted by stop")
        self.ensure_motion_ready()
        if self._robot_monitor is not None:
            return self._feedback_state(check_gripper_error=False)
        return self.get_state()

    def _on_arm_goal(self, future, generation: int) -> None:
        """Cancel an old goal even if its acceptance arrives after resume."""
        try:
            handle = future.result()
            with self._command_lock:
                if self._stopped or generation != self._motion_generation or not self._motion_active:
                    if handle.accepted:
                        handle.cancel_goal_async()
                    return
                if handle.accepted:
                    self._arm_goal = handle
        except Exception as exc:
            logger.error("Arm goal acceptance failed: %s", exc)

    def _joint_limit_vector(self, key):
        value = self._cfg.get(key)
        if value is None:
            return None
        limit = np.asarray(value, dtype=float)
        if limit.ndim == 0:
            limit = np.full(self.n_joints, float(limit))
        if limit.shape != (self.n_joints,) or not np.isfinite(limit).all() or np.any(limit <= 0):
            raise ValueError(f"{key} must be positive and finite: scalar or one value per joint")
        return limit

    def _bounded_trajectory_duration(self, q_start, q_target, requested):
        """Time-scale the entire quintic from fresh feedback, before vetting.

        For s(u)=10u^3-15u^4+6u^5, max s'=15/8 and
        max |s''|=10/sqrt(3). Preserve the path and zero endpoint derivatives;
        clipping individual positions or efforts would change both.
        These cap commanded motion; measured-motion safety remains active.
        """
        delta = np.abs(np.asarray(q_target) - np.asarray(q_start))
        duration = requested
        if self._trajectory_velocity_limit is not None:
            duration = max(duration, float(np.max(1.875 * delta / self._trajectory_velocity_limit)))
        if self._trajectory_acceleration_limit is not None:
            duration = max(duration, float(np.sqrt(np.max(
                (10.0 / np.sqrt(3.0)) * delta / self._trajectory_acceleration_limit))))
        return duration

    def _trajectory_goal(self, q_start, q_target, duration_s, tol, settle_timeout_s):
        from control_msgs.action import FollowJointTrajectory
        from control_msgs.msg import JointTolerance
        from trajectory_msgs.msg import JointTrajectoryPoint

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self._joints)
        # Two endpoints with q/dq/ddq select quintic JTC interpolation.
        # With zero endpoint derivatives this is exactly min_jerk(t/T),
        # without resetting the interpolator at every Python waypoint.
        for q, t in ((q_start, 0.0), (q_target, duration_s)):
            point = JointTrajectoryPoint()
            point.positions = q.tolist()
            point.velocities = [0.0] * self.n_joints
            point.accelerations = [0.0] * self.n_joints
            point.time_from_start = self._duration(t)
            goal.trajectory.points.append(point)
        goal.goal_tolerance = [JointTolerance(name=j, position=tol) for j in self._joints]
        goal.goal_time_tolerance = self._duration(settle_timeout_s)
        return goal

    def stream_to(self, q_target, duration_s, rate_hz=50.0, approve=None,
                  settle_tol=None, settle_timeout_s=None, preflight=None,
                  before_stream=None, feedback_guard=None, *, bias_compensate=False) -> bool:
        # bias_compensate is a simulator hint, not another hardware calibration
        # or target offset. The native controller receives the exact target.
        if self._trajectory_action is None:
            def guarded_start():
                if feedback_guard is not None:
                    feedback_guard(self.get_state())
                if before_stream is not None:
                    before_stream()

            def guarded_approve(previous, following, dt):
                if feedback_guard is not None:
                    feedback_guard(self.get_state())
                if approve is not None:
                    approve(previous, following, dt)

            try:
                result = super().stream_to(q_target, duration_s, rate_hz, guarded_approve,
                                           settle_tol, settle_timeout_s, preflight,
                                           guarded_start)
                if feedback_guard is not None:
                    feedback_guard(self.get_state())
                return result
            except Exception:
                self.stop()
                raise
        q_target = np.asarray(q_target, dtype=float).reshape(-1)
        if len(q_target) != self.n_joints or not np.isfinite(q_target).all():
            raise ValueError(f"Expected {self.n_joints} finite joint targets")
        tol = self.settle_tol if settle_tol is None else float(settle_tol)
        timeout = self.settle_timeout_s if settle_timeout_s is None else float(settle_timeout_s)
        if not all(np.isfinite(x) for x in (duration_s, rate_hz, tol, timeout)):
            raise ValueError("Motion duration, rate and tolerances must be finite")
        if duration_s <= 0 or rate_hz <= 0 or tol <= 0 or timeout < 0:
            raise ValueError("Motion duration, rate and position tolerance must be positive")
        if not self._motion_lock.acquire(blocking=False):
            raise RuntimeError("Another arm trajectory is already running")
        with self._command_lock:
            self._motion_generation += 1
            generation = self._motion_generation
            self._motion_active = True
            self._motion_error = None
        handle = None
        try:
            def observed_state():
                state = self._ensure_motion_ready(generation)
                if feedback_guard is not None:
                    feedback_guard(state)
                    # A callback may block, stop/reset, or observe a fault.
                    # Never authorize this generation from its earlier read.
                    self._ensure_motion_ready(generation)
                return state

            def check_before_stream():
                if before_stream is not None:
                    before_stream()
                self._ensure_motion_ready(generation)

            q_start = observed_state().q.copy()
            duration_s = self._bounded_trajectory_duration(q_start, q_target, duration_s)
            if preflight is not None:
                check_before_stream()
                preflight(q_start.copy(), duration_s)
                check_before_stream()
            if self._trajectory_client is None:
                raise RuntimeError("ROS2 trajectory action client is not connected")
            steps = max(2, int(np.ceil(duration_s * rate_hz)))
            dt = duration_s / steps
            q_prev = q_start
            # Every configuration is gated before any queued motion can
            # start. Runtime checks below still catch changes to the world.
            for i in range(1, steps + 1):
                self._ensure_motion_ready(generation)
                q_next = q_start + (q_target - q_start) * min_jerk(i / steps)
                if approve is not None:
                    approve(q_prev, q_next, dt)
                q_prev = q_next
            client = self._trajectory_client
            if not client.server_is_ready() and not client.wait_for_server(timeout_sec=2.0):
                raise RuntimeError(f"Arm trajectory action server unavailable: {self._trajectory_action}")
            goal = self._trajectory_goal(q_start, q_target, duration_s, tol, timeout)
            observed_state()
            check_before_stream()
            with self._command_lock:
                state = self._ensure_motion_ready(generation)
                # An external command during validation invalidates the
                # start state of the trajectory that was safety-approved.
                drift_limit = (PREFLIGHT_MAX_DRIFT_RAD if preflight is not None
                               else min(tol, 0.005))
                if np.max(np.abs(state.q - q_start)) > drift_limit:
                    raise RuntimeError("Arm moved while validating trajectory; replan from fresh feedback")
                start = time.monotonic()
                future = client.send_goal_async(goal)
                future.add_done_callback(lambda f: self._on_arm_goal(f, generation))
                self._last_cmd_q = q_target.copy()
            result_future = None
            q_plan_prev = q_start
            q_actual_prev = state.q.copy()
            actual_time_prev = state.t
            elapsed_prev = 0.0
            while True:
                state = observed_state()
                now = time.monotonic()
                elapsed = max(0.0, now - start)
                elapsed_dt = elapsed - elapsed_prev
                q_plan = q_start + (q_target - q_start) * min_jerk(min(elapsed / duration_s, 1.0))
                if approve is not None and elapsed >= dt and elapsed_dt > 0:
                    approve(q_plan_prev, q_plan, elapsed_dt)
                    # Fresh actual feedback must also stay safe. Skip a
                    # repeated measurement so a valid escape from an initial
                    # workspace/joint margin remains possible.
                    actual_dt = state.t - actual_time_prev
                    if actual_dt > 0 and np.max(np.abs(state.q - q_actual_prev)) > 1e-9:
                        approve(q_actual_prev, state.q, actual_dt)
                if elapsed >= dt:
                    q_plan_prev = q_plan
                    elapsed_prev = elapsed
                    if state.t > actual_time_prev:
                        q_actual_prev = state.q.copy()
                        actual_time_prev = state.t
                if result_future is None and future.done():
                    handle = future.result()
                    if not handle.accepted:
                        raise RuntimeError(f"Arm trajectory goal rejected by {self._trajectory_action}; "
                                           "check controller activation and trajectory validity")
                    result_future = handle.get_result_async()
                if result_future is not None and result_future.done():
                    result = result_future.result()
                    code = result.result.error_code
                    if result.status != 4 or code != 0:
                        names = {-1: "INVALID_GOAL", -2: "INVALID_JOINTS", -3: "OLD_HEADER_TIMESTAMP",
                                 -4: "PATH_TOLERANCE_VIOLATED", -5: "GOAL_TOLERANCE_VIOLATED"}
                        detail = result.result.error_string
                        # Preserve measured error before stop() replaces the
                        # target with a hold. The JTC timeout alone does not
                        # identify which joint missed the requested pose.
                        position_errors = np.abs(q_target - state.q)
                        worst = int(np.argmax(position_errors))
                        diagnostic = (
                            f"; largest measured position error {self._joints[worst]}="
                            f"{position_errors[worst]:.5f} rad (limit {tol:.5f} rad)"
                        )
                        raise RuntimeError(f"Arm trajectory failed: {names.get(code, code)} "
                                           f"(action status {result.status}) {detail}{diagnostic}".strip())
                    final = observed_state()
                    if approve is not None:
                        # The action result and JointState arrive on separate
                        # callbacks. Vet the newest measured pose as well as
                        # the snapshot checked before processing the result.
                        # A zero interval still runs all geometry checks.
                        approve(q_actual_prev, final.q, max(0.0, final.t - actual_time_prev))
                    error = float(np.max(np.abs(final.q - q_target)))
                    if error > tol:
                        raise RuntimeError(f"Arm trajectory reported success but feedback error "
                                           f"{error:.4f} rad exceeds {tol:.4f} rad")
                    return True
                if result_future is None and elapsed > 2.0:
                    raise RuntimeError("Arm trajectory goal acceptance timed out")
                if elapsed > duration_s + timeout + 0.5:
                    raise RuntimeError("Arm trajectory result timed out; controller stopped or unavailable")
                time.sleep(1.0 / rate_hz)
        except Exception as exc:
            with self._command_lock:
                self._motion_error = str(exc)
                if generation == self._motion_generation:
                    self.stop()
                    if isinstance(exc, MotionHalted):
                        # A harness halt abandons this action and permits
                        # redirect. An explicit STOP or fault stays latched.
                        self.resume()
            raise
        finally:
            with self._command_lock:
                if self._arm_goal is handle:
                    self._arm_goal = None
                self._motion_active = False
            self._motion_lock.release()

    def set_gripper(self, pos: float, effort: float = 1.0) -> None:
        self.ensure_motion_ready()
        generation = self._grip_generation
        if self._grip_joint is None or self._stopped:
            return
        if self._grip_mode == "action":
            if self._grip_client is None:
                raise RuntimeError("ROS2 gripper action client is not connected")
            from control_msgs.action import GripperCommand

            goal = GripperCommand.Goal()
            goal.command.position = float(pos)
            if not np.isfinite(pos) or not np.isfinite(effort):
                raise ValueError("Gripper position and effort must be finite")
            grip = self._cfg.get("gripper") or {}
            bounds = [grip.get("open_pos"), grip.get("closed_pos")]
            if all(v is not None for v in bounds) and not min(bounds) <= pos <= max(bounds):
                raise ValueError("Gripper position is outside the configured travel")
            goal.command.max_effort = float(np.clip(
                float(np.clip(effort, 0.0, 1.0)) * self._grip_max_effort,
                self._grip_min_effort, self._grip_max_effort))
            client, action_name = self._grip_client, self._grip_action
            if self._grasp_client is not None or self._grip_move_client is not None:
                with self._lock:
                    if self._last_gripper_t is None or time.monotonic() - self._last_gripper_t > self._state_timeout:
                        raise RuntimeError("Franka Gripper feedback is stale")
                    closing = pos < self._gripper_pos
                # A homed Hand can report slightly more than its nominal
                # 0.04 m finger travel. Opening to that endpoint must remain
                # a Move: both our old direction test and the generic Franka
                # GripperCommand server otherwise turn it into a force grasp.
                open_pos = grip.get("open_pos")
                fully_open = open_pos is not None and pos == float(open_pos)
                if self._grip_move_client is not None and (fully_open or not closing):
                    from franka_msgs.action import Move

                    goal = Move.Goal()
                    goal.width = 2.0 * float(pos)
                    goal.speed = float(self._cfg.get("ros_gripper_speed_m_s", 0.1))
                    client, action_name = self._grip_move_client, self._grip_move_action
                elif closing and self._grasp_client is not None:
                    from franka_msgs.action import Grasp

                    force = goal.command.max_effort
                    goal = Grasp.Goal()
                    goal.width = 2.0 * float(pos)
                    goal.speed = float(self._cfg.get("ros_gripper_speed_m_s", 0.1))
                    goal.force = force
                    goal.epsilon.inner = float(self._cfg.get("ros_gripper_epsilon_inner_m", 0.005))
                    goal.epsilon.outer = float(self._cfg.get("ros_gripper_epsilon_outer_m", 0.08))
                    client, action_name = self._grasp_client, self._grasp_action
            if not client.server_is_ready():
                # First call after bring-up: give discovery a moment rather
                # than dropping the goal on the floor.
                if not client.wait_for_server(timeout_sec=2.0):
                    raise RuntimeError(f"Gripper action server unavailable: {action_name}")
            with self._command_lock:
                if self._stopped or generation != self._grip_generation:
                    raise RuntimeError("Gripper command interrupted by stop")
                self.ensure_motion_ready()
                self._grip_error = None
                future = client.send_goal_async(goal)
                future.add_done_callback(
                    lambda future: self._on_gripper_goal(future, generation))
            if self._grip_wait_s > 0:
                deadline = time.monotonic() + self._grip_wait_s
                try:
                    handle = self._wait_gripper_future(future, deadline, generation)
                    if not handle.accepted:
                        raise RuntimeError("Gripper action goal was rejected")
                    result = self._wait_gripper_future(handle.get_result_async(), deadline, generation)
                    if result.status != 4 or not getattr(result.result, "success", True):
                        detail = getattr(result.result, "error", None)
                        # franka_gripper aborts BOTH a normal grasp miss and
                        # libfranka exceptions. Only the former has an empty
                        # error field. Never reinterpret transport/cancel/fault
                        # failures as contact, or clear an existing STOP.
                        if (client is self._grasp_client and result.status == 6
                                and getattr(result.result, "success", None) is False
                                and detail == ""):
                            with self._command_lock:
                                if self._stopped or generation != self._grip_generation:
                                    raise RuntimeError("Gripper command interrupted by stop")
                                self.ensure_motion_ready()
                                state = self.get_state()
                                if not state.gripper_valid:
                                    raise RuntimeError("Franka Gripper feedback is stale")
                                raise GraspFailed(
                                    f"Franka grasp did not secure the object: requested width "
                                    f"{goal.width:.4f} m, measured {2 * state.gripper_pos:.4f} m "
                                    f"(inner tolerance {goal.epsilon.inner:.4f} m, "
                                    f"outer {goal.epsilon.outer:.4f} m)")
                        raise RuntimeError(f"Gripper action failed on {action_name} "
                                           f"(status {result.status}): {detail or 'no driver detail'}")
                except GraspFailed:
                    raise
                except Exception:
                    self.stop()
                    raise
            return
        if self._grip_pub is None:
            return
        # effort is accepted for API parity; JointTrajectory gripper
        # controllers position-track and ignore per-point effort.
        if self._mode == "trajectory":
            self._grip_pub.publish(self._traj_msg([self._grip_joint], [float(pos)]))
        else:
            from std_msgs.msg import Float64MultiArray

            msg = Float64MultiArray()
            msg.data = [float(pos)]
            self._grip_pub.publish(msg)

    def _wait_gripper_future(self, future, deadline, generation):
        while True:
            if self._stopped or generation != self._grip_generation:
                raise RuntimeError("Gripper action interrupted by stop")
            if future.done():
                return future.result()
            if time.monotonic() >= deadline:
                raise RuntimeError("Gripper action timed out")
            time.sleep(0.01)

    def _on_gripper_goal(self, future, generation):
        try:
            handle = future.result()
            with self._command_lock:
                if self._stopped or generation != self._grip_generation:
                    if handle.accepted:
                        handle.cancel_goal_async()
                    return
                if not handle.accepted:
                    self._grip_error = "Gripper action goal was rejected"
                    return
                self._grip_goal = handle
        except Exception as exc:
            with self._command_lock:
                if generation == self._grip_generation and not self._stopped:
                    self._grip_error = f"Gripper action failed: {exc}"

    def stop(self) -> None:
        """Request measured arm hold and cancel the active gripper action."""
        with self._command_lock:
            self._stopped = True
            self._motion_generation += 1
            self._grip_generation += 1
            for handle, name in ((self._arm_goal, "Arm"), (self._grip_goal, "Gripper")):
                if handle is not None:
                    try:
                        handle.cancel_goal_async()
                    except Exception as exc:
                        logger.error("%s cancel failed: %s", name, exc)
            if not self.connected or self._cmd_pub is None:
                return
            try:
                if self._mode == "trajectory":
                    with self._lock:
                        if self._last_state_t is None or time.monotonic() - self._last_state_t > self._state_timeout:
                            raise RuntimeError("Cannot form a measured hold from stale arm feedback")
                        measured = [self._pos[j] for j in self._joints]
                    msg = self._traj_msg(self._joints, measured)
                    msg.points[0].velocities = [0.0] * self.n_joints
                    msg.points[0].accelerations = [0.0] * self.n_joints
                    self._cmd_pub.publish(msg)
                elif self._last_cmd_q is not None:
                    from std_msgs.msg import Float64MultiArray

                    msg = Float64MultiArray()
                    msg.data = self._last_cmd_q.tolist()
                    self._cmd_pub.publish(msg)
            except Exception as e:  # noqa: BLE001 - stop must never raise
                logger.error("ros2 soft stop failed to publish: %s", e)

    def check_resume_ready(self) -> None:
        """Read-only reset preflight, including actual controller activation."""
        if not self.connected or self._motion_active:
            raise RuntimeError("Arm disconnected or still executing a motion")
        if self._robot_monitor is not None:
            self._robot_monitor.ensure_ready()
        state = self._feedback_state(check_gripper_error=True)
        if self._grip_joint and not state.gripper_valid:
            raise RuntimeError("Gripper feedback is stale; check Franka gripper bringup")
        if self._trajectory_action:
            from controller_manager_msgs.srv import ListControllers

            service = self._resolve(str(self._cfg.get(
                "ros_controller_manager", "controller_manager"))) + "/list_controllers"
            client = self._node.create_client(ListControllers, service)
            future = None
            try:
                if not client.wait_for_service(timeout_sec=1.0):
                    raise RuntimeError("Controller manager unavailable; relaunch ROS bringup")
                future = client.call_async(ListControllers.Request())
                deadline = time.monotonic() + 2.0
                while not future.done() and time.monotonic() < deadline:
                    time.sleep(0.01)
                if not future.done():
                    raise RuntimeError("Controller readiness check timed out")
                name = self._trajectory_action.rstrip("/").split("/")[-2]
                active = any(c.name == name and c.state == "active"
                             for c in future.result().controller)
                if not active:
                    raise RuntimeError(f"Controller {name} is inactive; relaunch ROS bringup")
            finally:
                if future is not None and not future.done():
                    future.cancel()
                self._node.destroy_client(client)
        if self._robot_monitor is not None:
            self._robot_monitor.ensure_ready()
        state = self._feedback_state(check_gripper_error=True)
        if self._grip_joint and not state.gripper_valid:
            raise RuntimeError("Gripper feedback is stale; check Franka gripper bringup")

    def resume(self) -> None:
        with self._command_lock:
            self._stopped = False
