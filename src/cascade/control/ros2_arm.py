"""Any ROS2 robot as a cascade arm (`type: ros2`).

The transport speaks the two interfaces every ros2_control deployment (and
the rosclaw bridge) already exposes, so "adding a robot" is a profile file
naming its topics -- no Python:

  state    sensor_msgs/JointState   subscribed with SENSOR_DATA QoS
           (best-effort matches both best-effort and reliable publishers;
           a RELIABLE subscription to a best-effort driver never pairs in
           DDS and reads as "the arm publishes nothing").
  command  trajectory_msgs/JointTrajectory, one point per streamed waypoint
           (`ros_command: trajectory`, the joint_trajectory_controller
           path), or std_msgs/Float64MultiArray for a
           forward_position_controller (`ros_command: position`).
  gripper  same publisher family on `ros_gripper_topic` (default), or a
           control_msgs/GripperCommand ACTION (`ros_gripper: action`,
           `ros_gripper_action: <name>`) for drivers like franka_gripper
           and robotiq that expose no trajectory topic for the fingers.
           `ros_gripper_state_topic` adds a second JointState subscription
           when the fingers are published separately from the arm.

           50 Hz STREAMING TRADE (measured pattern, rosclaw brief AVOID#4):
           a single-point JointTrajectory per waypoint makes a JTC restart
           its interpolation every 20 ms -- functional (each trajectory
           legally supersedes the last; time_from_start is sec+nanosec so
           sub-second points never truncate to 0) but wasteful, and some
           controller configs rate-limit replacement goals. For a robot
           whose bringup offers a forward/JointGroupPositionController,
           prefer `ros_command: position` for the streaming path; keep
           `trajectory` (the default) for stock bringups that ship only a
           JTC. Point-to-point moves are fine on either.

Joints are addressed BY NAME, never by index: a JointState's order is
whatever the driver publishes (arm + gripper interleaved, alphabetical,
split across messages), and any index would silently shift. The profile's
`ros_joints` list defines this framework's q order; each incoming message
updates only the names it carries.

`stop()` in trajectory mode publishes an EMPTY JointTrajectory -- the
ros2_control semantic for "cancel the current trajectory and hold here" --
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
from ..types import RobotState
from .arm_base import ArmBase

logger = logging.getLogger(__name__)


class Ros2Arm(ArmBase):
    def __init__(self, cfg: Cfg):
        self._cfg = cfg
        self.n_joints = int(cfg.get("n_joints", 6))
        self.settle_tol = float(cfg.get("settle_tol", 0.03))
        self.settle_timeout_s = float(cfg.get("settle_timeout_s", 3.0))

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
        self._last_state_t: float | None = None
        self._gripper_pos = 0.0
        self._gripper_valid = False

    def _resolve(self, topic: str) -> str:
        """Namespace-prefix relative topics; absolute (`/x`) pass through."""
        if topic.startswith("/"):
            return topic
        return f"/{self._ns}/{topic}" if self._ns else f"/{topic}"

    # ── lifecycle ────────────────────────────────────────────────────────

    def connect(self) -> None:
        if self.connected:
            return
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
        if self._grip_mode == "action":
            try:
                from control_msgs.action import GripperCommand
                from rclpy.action import ActionClient
            except ImportError as e:
                raise RuntimeError(
                    "ros_gripper: action needs control_msgs (ros-<distro>-control-msgs)"
                ) from e
            self._grip_client = ActionClient(self._node, GripperCommand, self._grip_action)

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
        self.connected = False

    # ── feedback ─────────────────────────────────────────────────────────

    def _on_joint_state(self, msg) -> None:
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                self._pos[str(name)] = float(pos)
            self._last_state_t = time.monotonic()
            if self._grip_joint and self._grip_joint in self._pos:
                self._gripper_pos = self._pos[self._grip_joint]
                self._gripper_valid = True

    def get_state(self) -> RobotState:
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
            return RobotState(
                q=q, gripper_pos=self._gripper_pos,
                gripper_valid=self._gripper_valid, t=last_t,
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
        if self._stopped:
            return
        q = np.asarray(q, dtype=float).reshape(-1)[: self.n_joints]
        if self._mode == "trajectory":
            self._cmd_pub.publish(self._traj_msg(self._joints, q.tolist()))
        else:
            from std_msgs.msg import Float64MultiArray

            msg = Float64MultiArray()
            msg.data = q.tolist()
            self._cmd_pub.publish(msg)
        self._last_cmd_q = q.copy()

    def set_gripper(self, pos: float, effort: float = 1.0) -> None:
        if self._grip_joint is None or self._stopped:
            return
        if self._grip_mode == "action":
            if self._grip_client is None:
                return
            from control_msgs.action import GripperCommand

            goal = GripperCommand.Goal()
            goal.command.position = float(pos)
            goal.command.max_effort = float(np.clip(effort, 0.0, 1.0)) * self._grip_max_effort
            if not self._grip_client.server_is_ready():
                # First call after bring-up: give discovery a moment rather
                # than dropping the goal on the floor.
                self._grip_client.wait_for_server(timeout_sec=2.0)
            # Fire-and-forget like the topic path: the skill layer already
            # settles on feedback (joint_states) and stall detection.
            self._grip_client.send_goal_async(goal)
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

    def stop(self) -> None:
        """Soft stop. Trajectory mode: an EMPTY JointTrajectory is the
        ros2_control cancel-and-hold; position mode re-asserts the last
        target (a forward controller has no cancel semantics)."""
        self._stopped = True
        if not self.connected or self._cmd_pub is None:
            return
        try:
            if self._mode == "trajectory":
                from trajectory_msgs.msg import JointTrajectory

                self._cmd_pub.publish(JointTrajectory())
            elif self._last_cmd_q is not None:
                from std_msgs.msg import Float64MultiArray

                msg = Float64MultiArray()
                msg.data = self._last_cmd_q.tolist()
                self._cmd_pub.publish(msg)
        except Exception as e:  # noqa: BLE001 - stop must never raise
            logger.error("ros2 soft stop failed to publish: %s", e)

    def resume(self) -> None:
        self._stopped = False
