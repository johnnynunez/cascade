"""Ros2Arm (`type: ros2`) against a stubbed rclpy stack.

CI has no ROS2 distro, so the whole rclpy/message surface is faked in
sys.modules (the conftest convention: never importorskip an optional dep a
stub can stand in for). The stub is behavioural, not cosmetic: publishes
are recorded, the JointState subscription is driven by hand, and the
spin-thread lifecycle runs for real.
"""

from __future__ import annotations

import sys
import threading
import time
import types

import numpy as np
import pytest

from cascade.config import load_demo_config, load_profile


# ── the rclpy stub ────────────────────────────────────────────────────────

class _FakeDuration:
    def __init__(self, sec=0, nanosec=0):
        self.sec, self.nanosec = sec, nanosec


class _FakeJointState:
    def __init__(self):
        self.name: list[str] = []
        self.position: list[float] = []


class _FakeJTPoint:
    def __init__(self):
        self.positions: list[float] = []
        self.time_from_start = _FakeDuration()


class _FakeJointTrajectory:
    def __init__(self):
        self.joint_names: list[str] = []
        self.points: list[_FakeJTPoint] = []


class _FakeFloat64MultiArray:
    def __init__(self):
        self.data: list[float] = []


class _FakePublisher:
    def __init__(self, msg_type, topic):
        self.msg_type = msg_type
        self.topic = topic
        self.published: list = []

    def publish(self, msg):
        self.published.append(msg)


class _FakeSubscription:
    def __init__(self, msg_type, topic, cb, qos):
        self.msg_type, self.topic, self.cb, self.qos = msg_type, topic, cb, qos


class _FakeNode:
    instances: list["_FakeNode"] = []

    def __init__(self, name):
        self.name = name
        self.subs: list[_FakeSubscription] = []
        self.pubs: list[_FakePublisher] = []
        self.destroyed = False
        _FakeNode.instances.append(self)

    def create_subscription(self, msg_type, topic, cb, qos):
        sub = _FakeSubscription(msg_type, topic, cb, qos)
        self.subs.append(sub)
        return sub

    def create_publisher(self, msg_type, topic, depth):
        pub = _FakePublisher(msg_type, topic)
        self.pubs.append(pub)
        return pub

    def destroy_node(self):
        self.destroyed = True


class _FakeExecutor:
    def __init__(self):
        self._shutdown = threading.Event()
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def spin(self):
        self._shutdown.wait()

    def shutdown(self, timeout_sec=None):
        self._shutdown.set()


class _RclpyState:
    ok_flag = False


def _install_ros2_stub(monkeypatch):
    _FakeNode.instances = []
    rclpy = types.ModuleType("rclpy")
    _RclpyState.ok_flag = False

    def _init():
        _RclpyState.ok_flag = True

    def _ok():
        return _RclpyState.ok_flag

    def _shutdown():
        _RclpyState.ok_flag = False

    rclpy.init = _init
    rclpy.ok = _ok
    rclpy.shutdown = _shutdown

    executors = types.ModuleType("rclpy.executors")
    executors.SingleThreadedExecutor = _FakeExecutor
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = _FakeNode
    qos_mod = types.ModuleType("rclpy.qos")
    qos_mod.qos_profile_sensor_data = object()

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.JointState = _FakeJointState
    trajectory_msgs = types.ModuleType("trajectory_msgs")
    trajectory_msgs_msg = types.ModuleType("trajectory_msgs.msg")
    trajectory_msgs_msg.JointTrajectory = _FakeJointTrajectory
    trajectory_msgs_msg.JointTrajectoryPoint = _FakeJTPoint
    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Float64MultiArray = _FakeFloat64MultiArray
    builtin_ifaces = types.ModuleType("builtin_interfaces")
    builtin_ifaces_msg = types.ModuleType("builtin_interfaces.msg")
    builtin_ifaces_msg.Duration = _FakeDuration

    for name, mod in {
        "rclpy": rclpy, "rclpy.executors": executors, "rclpy.node": node_mod,
        "rclpy.qos": qos_mod,
        "sensor_msgs": sensor_msgs, "sensor_msgs.msg": sensor_msgs_msg,
        "trajectory_msgs": trajectory_msgs,
        "trajectory_msgs.msg": trajectory_msgs_msg,
        "std_msgs": std_msgs, "std_msgs.msg": std_msgs_msg,
        "builtin_interfaces": builtin_ifaces,
        "builtin_interfaces.msg": builtin_ifaces_msg,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)


@pytest.fixture
def ros2_stub(monkeypatch):
    _install_ros2_stub(monkeypatch)
    yield


_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def _mk_arm(**over):
    from cascade.config import Cfg
    from cascade.control.arm_base import make_arm

    data = {
        "type": "ros2", "name": "left", "n_joints": 5,
        "ros_namespace": "so101",
        "ros_joints": list(_JOINTS),
        "ros_gripper_joint": "gripper",
        "ros_gripper_topic": "gripper_trajectory_controller/joint_trajectory",
        "wait_state_s": 2.0,
        "gripper": {"open_pos": 0.6, "closed_pos": -0.15, "max_width_m": 0.055},
    }
    data.update(over)
    return make_arm(Cfg(data))


def _feed_state(node, q, names=None, gripper=None):
    msg = _FakeJointState()
    msg.name = list(names or _JOINTS)
    msg.position = [float(x) for x in q]
    if gripper is not None:
        msg.name.append("gripper")
        msg.position.append(float(gripper))
    for sub in node.subs:
        sub.cb(msg)


def _connect(arm):
    """connect() blocks until the first JointState; feed it from a thread."""
    def feeder():
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not _FakeNode.instances:
            time.sleep(0.01)
        for _ in range(50):
            if _FakeNode.instances:
                _feed_state(_FakeNode.instances[-1], np.zeros(5), gripper=0.6)
                if getattr(arm, "connected", False):
                    return
            time.sleep(0.02)

    t = threading.Thread(target=feeder, daemon=True)
    t.start()
    arm.connect()
    t.join(timeout=1.0)


def test_make_arm_dispatches_ros2(ros2_stub):
    arm = _mk_arm()
    from cascade.control.ros2_arm import Ros2Arm

    assert isinstance(arm, Ros2Arm)


def test_topics_are_namespaced(ros2_stub):
    arm = _mk_arm()
    _connect(arm)
    try:
        node = _FakeNode.instances[-1]
        assert node.subs[0].topic == "/so101/joint_states"
        cmd = [p for p in node.pubs
               if p.topic == "/so101/joint_trajectory_controller/joint_trajectory"]
        assert cmd, [p.topic for p in node.pubs]
        # absolute topics pass through untouched
    finally:
        arm.disconnect()


def test_absolute_topic_passthrough(ros2_stub):
    arm = _mk_arm(ros_state_topic="/robot/joint_states")
    try:
        assert arm._state_topic == "/robot/joint_states"
    finally:
        pass


def test_joint_order_is_by_name_never_by_index(ros2_stub):
    """The driver publishing a REVERSED (or interleaved) JointState must not
    corrupt q: names map to the profile's ros_joints order."""
    arm = _mk_arm()
    _connect(arm)
    try:
        node = _FakeNode.instances[-1]
        q_named = {n: i * 0.1 for i, n in enumerate(_JOINTS)}
        rev = list(reversed(_JOINTS))
        _feed_state(node, [q_named[n] for n in rev], names=rev, gripper=0.3)
        q = arm.get_state().q
        np.testing.assert_allclose(q, [q_named[n] for n in _JOINTS])
        assert arm.get_state().gripper_pos == pytest.approx(0.3)
    finally:
        arm.disconnect()


def test_connect_times_out_without_publisher(ros2_stub):
    arm = _mk_arm(wait_state_s=0.3)
    with pytest.raises(RuntimeError, match="no JointState"):
        arm.connect()
    assert not arm.connected


def test_send_joint_target_publishes_trajectory(ros2_stub):
    arm = _mk_arm()
    _connect(arm)
    try:
        node = _FakeNode.instances[-1]
        cmd = next(p for p in node.pubs
                   if p.topic.endswith("joint_trajectory_controller/joint_trajectory"))
        q = np.array([0.1, -0.2, 0.3, 1.0, -0.5])
        arm.send_joint_target(q)
        msg = cmd.published[-1]
        assert msg.joint_names == _JOINTS
        np.testing.assert_allclose(msg.points[0].positions, q)
    finally:
        arm.disconnect()


def test_position_mode_publishes_float_array(ros2_stub):
    arm = _mk_arm(ros_command="position",
                  ros_command_topic="forward_position_controller/commands")
    _connect(arm)
    try:
        node = _FakeNode.instances[-1]
        cmd = next(p for p in node.pubs
                   if p.topic.endswith("forward_position_controller/commands"))
        arm.send_joint_target(np.array([0.5, 0, 0, 0, 0]))
        assert cmd.published[-1].data[0] == pytest.approx(0.5)
    finally:
        arm.disconnect()


def test_stop_publishes_measured_hold_and_suppresses(ros2_stub):
    """JTC accepts a named measured hold, not an empty trajectory; further
    waypoints are dropped until resume() (soft-stop contract)."""
    arm = _mk_arm()
    _connect(arm)
    try:
        node = _FakeNode.instances[-1]
        cmd = next(p for p in node.pubs
                   if p.topic.endswith("joint_trajectory_controller/joint_trajectory"))
        arm.stop()
        msg = cmd.published[-1]
        assert msg.joint_names == arm._joints
        assert msg.points[0].positions == arm.get_state().q.tolist()
        assert msg.points[0].velocities == [0.0] * 5
        n = len(cmd.published)
        arm.send_joint_target(np.zeros(5))   # suppressed
        assert len(cmd.published) == n
        arm.resume()
        arm.send_joint_target(np.zeros(5))
        assert len(cmd.published) == n + 1
    finally:
        arm.disconnect()


def test_stale_state_raises(ros2_stub, monkeypatch):
    from cascade.control import ros2_arm

    clock = types.SimpleNamespace(now=0.0)
    monkeypatch.setattr(ros2_arm, "time", types.SimpleNamespace(
        monotonic=lambda: clock.now,
        sleep=lambda _: pytest.fail("synchronous JointState setup waited"),
    ))
    add_node = _FakeExecutor.add_node

    def add_node_with_state(executor, node):
        add_node(executor, node)
        _feed_state(node, np.zeros(5), gripper=0.6)

    # Supply feedback before connect checks freshness, independent of scheduling.
    monkeypatch.setattr(_FakeExecutor, "add_node", add_node_with_state)
    arm = _mk_arm(state_timeout_s=0.05)
    try:
        arm.connect()
        assert arm.get_state().t == 0.0
        clock.now = 0.05
        assert arm.get_state().t == 0.0
        clock.now = 0.051
        with pytest.raises(RuntimeError, match="stale"):
            arm.get_state()
    finally:
        arm.disconnect()


def test_gripper_command_uses_its_own_controller(ros2_stub):
    arm = _mk_arm()
    _connect(arm)
    try:
        node = _FakeNode.instances[-1]
        grip = next(p for p in node.pubs
                    if p.topic.endswith("gripper_trajectory_controller/joint_trajectory"))
        arm.set_gripper(0.6)
        msg = grip.published[-1]
        assert msg.joint_names == ["gripper"]
        assert msg.points[0].positions == [0.6]
    finally:
        arm.disconnect()


def test_ros_joints_must_match_n_joints(ros2_stub):
    with pytest.raises(ValueError, match="ros_joints"):
        _mk_arm(ros_joints=["a", "b"])


# ── GripperCommand action path (franka_gripper, robotiq) ─────────────────

class _FakeGoalHandleFuture:
    def __init__(self, accepted=True):
        self.handle = types.SimpleNamespace(accepted=accepted, cancel_goal_async=self.cancel,
                                            get_result_async=self.get_result_async)
        self.cancelled = False

    def done(self):
        return True

    def get_result_async(self):
        return types.SimpleNamespace(done=lambda: True, result=lambda: types.SimpleNamespace(
            status=4, result=types.SimpleNamespace(success=True)))

    def cancel(self):
        self.cancelled = True

    def result(self):
        return self.handle

    def add_done_callback(self, callback):
        callback(self)


class _FakeActionClient:
    instances: list["_FakeActionClient"] = []

    def __init__(self, node, action_type, name):
        self.node, self.action_type, self.name = node, action_type, name
        self.goals: list = []
        self.ready = True
        _FakeActionClient.instances.append(self)

    def server_is_ready(self):
        return self.ready

    def wait_for_server(self, timeout_sec=None):
        return self.ready

    def send_goal_async(self, goal):
        self.goals.append(goal)
        return _FakeGoalHandleFuture()


class _FakeGripperCommandMsg:
    def __init__(self):
        self.position = 0.0
        self.max_effort = 0.0


class _FakeGripperGoal:
    def __init__(self):
        self.command = _FakeGripperCommandMsg()


class _FakeGripperCommand:
    Goal = _FakeGripperGoal


@pytest.fixture
def gripper_action_stub(ros2_stub, monkeypatch):
    _FakeActionClient.instances = []
    action_mod = types.ModuleType("rclpy.action")
    action_mod.ActionClient = _FakeActionClient
    control_msgs = types.ModuleType("control_msgs")
    control_msgs_action = types.ModuleType("control_msgs.action")
    control_msgs_action.GripperCommand = _FakeGripperCommand
    franka_msgs = types.ModuleType("franka_msgs")
    franka_actions = types.ModuleType("franka_msgs.action")
    franka_actions.Grasp = types.SimpleNamespace(Goal=lambda: types.SimpleNamespace(
        width=0.0, speed=0.0, force=0.0, epsilon=types.SimpleNamespace(inner=0.0, outer=0.0)))
    franka_actions.Move = types.SimpleNamespace(Goal=lambda: types.SimpleNamespace(width=0.0, speed=0.0))
    for name, mod in {"rclpy.action": action_mod, "control_msgs": control_msgs,
                      "control_msgs.action": control_msgs_action,
                      "franka_msgs": franka_msgs, "franka_msgs.action": franka_actions}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    yield


def _mk_fr3_like(**over):
    data = {
        "ros_namespace": "", "ros_command_topic": "fr3_arm_controller/joint_trajectory",
        "ros_gripper": "action", "ros_gripper_action": "franka_gripper/gripper_action",
        "ros_gripper_joint": "gripper", "ros_gripper_state_topic": "franka_gripper/joint_states",
        "gripper": {"open_pos": 0.04, "closed_pos": 0.0, "max_width_m": 0.08, "max_effort_n": 70.0},
    }
    data.update(over)
    data.pop("ros_gripper_topic", None)
    return _mk_arm(**data)


def test_gripper_action_sends_goal_with_position_and_scaled_effort(gripper_action_stub):
    """`ros_gripper: action`: no trajectory publisher for the fingers; a
    GripperCommand goal carries the finger joint value and effort*max_effort_n."""
    arm = _mk_fr3_like()
    _connect(arm)
    try:
        node = _FakeNode.instances[-1]
        assert not [p for p in node.pubs if "gripper" in p.topic], "no gripper topic publisher in action mode"
        client = _FakeActionClient.instances[-1]
        assert client.name == "/franka_gripper/gripper_action"
        arm.set_gripper(0.0, effort=0.5)
        goal = client.goals[-1]
        assert goal.command.position == pytest.approx(0.0)
        assert goal.command.max_effort == pytest.approx(35.0)
        arm.set_gripper(0.04)
        assert client.goals[-1].command.position == pytest.approx(0.04)
        assert client.goals[-1].command.max_effort == pytest.approx(70.0)
    finally:
        arm.disconnect()


def test_gripper_state_topic_adds_a_second_joint_state_subscription(gripper_action_stub):
    """franka_gripper publishes the fingers on its own joint_states; the
    finger value must still land in RobotState.gripper_pos."""
    arm = _mk_fr3_like()
    _connect(arm)
    try:
        node = _FakeNode.instances[-1]
        topics = sorted(s.topic for s in node.subs)
        assert topics == ["/franka_gripper/joint_states", "/joint_states"]
        # feed the arm joints WITHOUT the gripper, then the gripper alone
        _feed_state(node, np.zeros(5))
        msg = _FakeJointState(); msg.name = ["gripper"]; msg.position = [0.031]
        next(s for s in node.subs if s.topic == "/franka_gripper/joint_states").cb(msg)
        st = arm.get_state()
        assert st.gripper_valid and st.gripper_pos == pytest.approx(0.031)
    finally:
        arm.disconnect()


def test_native_franka_close_uses_full_width_and_waits_for_result(gripper_action_stub):
    arm = _mk_fr3_like(ros_gripper_grasp_action="franka_gripper/grasp", ros_gripper_wait_s=0.5)
    _connect(arm)
    try:
        msg = _FakeJointState(); msg.name = ["gripper"]; msg.position = [0.04]
        arm._on_joint_state(msg)
        arm.set_gripper(0.01, effort=0.5)
        move, grasp = _FakeActionClient.instances
        assert not move.goals
        goal = grasp.goals[-1]
        assert goal.width == pytest.approx(0.02)
        assert goal.force == pytest.approx(35)
        assert goal.speed == pytest.approx(0.1)
        assert goal.epsilon.outer == pytest.approx(0.08)
        msg.position = [0.02]; arm._on_joint_state(msg)
        arm.set_gripper(0.04)
        assert move.goals[-1].command.position == pytest.approx(0.04)
    finally:
        arm.disconnect()


@pytest.mark.parametrize("measured", [0.04010147601366043, 0.04, 0.0])
def test_native_franka_open_uses_move_even_when_feedback_exceeds_nominal(gripper_action_stub, measured):
    arm = _mk_fr3_like(ros_gripper_grasp_action="franka_gripper/grasp",
                      ros_gripper_move_action="franka_gripper/move", ros_gripper_wait_s=0.5)
    _connect(arm)
    try:
        msg = _FakeJointState(); msg.name = ["gripper"]; msg.position = [measured]
        arm._on_joint_state(msg)
        arm.set_gripper(0.04, effort=0.8)
        generic, grasp, move = _FakeActionClient.instances
        assert not generic.goals and not grasp.goals
        assert move.goals[-1].width == pytest.approx(0.08)
        assert move.goals[-1].speed == pytest.approx(0.1)
    finally:
        arm.disconnect()
    assert arm._grip_move_client is None


@pytest.mark.parametrize("close_position", [0.0, 0.01, 0.019999])
def test_native_franka_partial_open_then_force_close(gripper_action_stub, close_position):
    arm = _mk_fr3_like(ros_gripper_grasp_action="franka_gripper/grasp",
                      ros_gripper_move_action="franka_gripper/move", ros_gripper_wait_s=0.5)
    _connect(arm)
    try:
        msg = _FakeJointState(); msg.name = ["gripper"]; msg.position = [0.005]
        arm._on_joint_state(msg)
        arm.set_gripper(0.02)
        generic, grasp, move = _FakeActionClient.instances
        assert move.goals[-1].width == pytest.approx(0.04)
        msg.position = [0.02]; arm._on_joint_state(msg)
        arm.set_gripper(close_position, effort=0.5)
        assert grasp.goals[-1].width == pytest.approx(2 * close_position)
        assert grasp.goals[-1].force == pytest.approx(35)
        assert len(move.goals) == 1 and not generic.goals
    finally:
        arm.disconnect()


def test_native_franka_move_does_not_fallback_when_server_unavailable(gripper_action_stub):
    arm = _mk_fr3_like(ros_gripper_grasp_action="franka_gripper/grasp",
                      ros_gripper_move_action="franka_gripper/move", ros_gripper_wait_s=0.5)
    _connect(arm)
    try:
        msg = _FakeJointState(); msg.name = ["gripper"]; msg.position = [0.04010147601366043]
        arm._on_joint_state(msg)
        generic, grasp, move = _FakeActionClient.instances
        move.ready = False
        with pytest.raises(RuntimeError, match="unavailable: /franka_gripper/move"):
            arm.set_gripper(0.04)
        assert not generic.goals and not grasp.goals and not move.goals
    finally:
        arm.disconnect()


def test_native_franka_move_timeout_cancels_and_stops(gripper_action_stub, monkeypatch):
    arm = _mk_fr3_like(ros_gripper_grasp_action="franka_gripper/grasp",
                      ros_gripper_move_action="franka_gripper/move", ros_gripper_wait_s=0.02)
    _connect(arm)
    try:
        msg = _FakeJointState(); msg.name = ["gripper"]; msg.position = [0.0]
        arm._on_joint_state(msg)
        future = _FakeGoalHandleFuture()
        future.handle.get_result_async = lambda: types.SimpleNamespace(done=lambda: False)
        monkeypatch.setattr(arm._grip_move_client, "send_goal_async", lambda _: future)
        with pytest.raises(RuntimeError, match="timed out"):
            arm.set_gripper(0.04)
        assert future.cancelled and arm._stopped
    finally:
        arm.disconnect()


def test_gripper_wait_timeout_cancels_and_stops(gripper_action_stub, monkeypatch):
    arm = _mk_fr3_like(ros_gripper_wait_s=0.02)
    _connect(arm)
    try:
        future = _FakeGoalHandleFuture()
        future.handle.get_result_async = lambda: types.SimpleNamespace(done=lambda: False)
        monkeypatch.setattr(_FakeActionClient.instances[-1], "send_goal_async", lambda _: future)
        with pytest.raises(RuntimeError, match="timed out"):
            arm.set_gripper(0.01)
        assert future.cancelled and arm._stopped
    finally:
        arm.disconnect()


def test_gripper_action_requires_its_two_keys(ros2_stub):
    with pytest.raises(ValueError, match="ros_gripper_action"):
        _mk_arm(ros_gripper="action", ros_gripper_action=None)
    with pytest.raises(ValueError, match="topic|action"):
        _mk_arm(ros_gripper="service")


def test_fr3_profile_is_a_complete_franka_ros2_wiring():
    """fr3.yaml must name franka_ros2's actual interfaces, with 7 DoF and a
    Franka-Hand jaw in FINGER units (0.04 m per finger = 0.08 m width)."""
    prof = load_profile("arms", "fr3").as_dict()
    assert prof["type"] == "ros2" and prof["n_joints"] == 7
    assert prof["ros_command_topic"] == "fr3_arm_controller/joint_trajectory"
    assert prof["ros_trajectory_action"] == "fr3_arm_controller/follow_joint_trajectory"
    assert prof["ros_robot_state_topic"] == "franka_robot_state_broadcaster/robot_state"
    assert prof["ros_state_topic"] == "franka/joint_states"
    assert prof["ros_joints"] == [f"fr3_joint{i}" for i in range(1, 8)]
    assert prof["ros_gripper"] == "action"
    assert prof["ros_gripper_action"] == "franka_gripper/gripper_action"
    assert prof["ros_gripper_move_action"] == "franka_gripper/move"
    assert prof["ros_gripper_grasp_action"] == "franka_gripper/grasp"
    assert prof["ros_gripper_joint"] == "fr3_finger_joint1"
    assert prof["gripper"]["open_pos"] == pytest.approx(0.04)
    assert prof["gripper"]["max_width_m"] == pytest.approx(0.08)
    assert prof["tool_axis_order"] == "third_open_down"
    cfg = load_demo_config(camera="mock", arm="fr3_mock", llm="mock")
    assert cfg.arm.type == "mock" and cfg.safety.workspace.max[0] == pytest.approx(0.80)


def test_gripper_feedback_cannot_refresh_stale_arm(gripper_action_stub):
    arm = _mk_fr3_like(state_timeout_s=0.05)
    _connect(arm)
    try:
        time.sleep(0.07)
        msg = _FakeJointState(); msg.name = ["gripper"]; msg.position = [0.03]
        arm._on_joint_state(msg)
        with pytest.raises(RuntimeError, match="stale"):
            arm.get_state()
    finally:
        arm.disconnect()


def test_partial_joint_update_cannot_refresh_other_joints(ros2_stub):
    arm = _mk_arm(state_timeout_s=0.05)
    _connect(arm)
    try:
        time.sleep(0.07)
        msg = _FakeJointState(); msg.name = [arm._joints[0]]; msg.position = [0.1]
        arm._on_joint_state(msg)
        with pytest.raises(RuntimeError, match="stale"):
            arm.get_state()
    finally:
        arm.disconnect()


def test_franka_force_floor_and_unavailable_action(gripper_action_stub):
    arm = _mk_fr3_like(gripper={"open_pos": 0.04, "closed_pos": 0.0,
                              "max_effort_n": 70.0, "min_effort_n": 20.0})
    _connect(arm)
    try:
        client = _FakeActionClient.instances[-1]
        arm.set_gripper(0.01, effort=0.1)
        assert client.goals[-1].command.position == 0.01
        assert client.goals[-1].command.max_effort == 20.0
        client.ready = False
        with pytest.raises(RuntimeError, match="unavailable"):
            arm.set_gripper(0.04)
        assert len(client.goals) == 1
    finally:
        arm.disconnect()


def test_stop_cancels_late_gripper_acceptance_even_after_resume(gripper_action_stub):
    arm = _mk_fr3_like()
    _connect(arm)
    try:
        generation = arm._grip_generation
        arm.stop()
        arm.resume()
        current = _FakeGoalHandleFuture()
        arm._on_gripper_goal(current, arm._grip_generation)
        future = _FakeGoalHandleFuture()
        arm._on_gripper_goal(future, generation)
        assert future.cancelled
        assert arm._grip_goal is current.handle
        rejected = _FakeGoalHandleFuture(accepted=False)
        arm._on_gripper_goal(rejected, arm._grip_generation)
        with pytest.raises(RuntimeError, match="rejected"):
            arm.get_state()
    finally:
        arm.disconnect()


def test_stop_during_gripper_discovery_prevents_goal(gripper_action_stub, monkeypatch):
    arm = _mk_fr3_like()
    _connect(arm)
    try:
        client = _FakeActionClient.instances[-1]
        client.ready = False

        def discover(timeout_sec):
            arm.stop()
            arm.resume()
            return True

        monkeypatch.setattr(client, "wait_for_server", discover)
        with pytest.raises(RuntimeError, match="interrupted by stop"):
            arm.set_gripper(0.01)
        assert not client.goals
    finally:
        arm.disconnect()


def test_gripper_acceptance_and_stop_are_atomic_across_threads(gripper_action_stub):
    arm = _mk_fr3_like()
    _connect(arm)
    inspected = threading.Event()
    release = threading.Event()
    stopped = threading.Event()
    cancelled = threading.Event()

    class PausedHandle:
        @property
        def accepted(self):
            inspected.set()
            assert release.wait(1.0)
            return True

        def cancel_goal_async(self):
            cancelled.set()

    handle = PausedHandle()
    future = types.SimpleNamespace(result=lambda: handle)
    callback = threading.Thread(target=arm._on_gripper_goal, args=(future, arm._grip_generation))
    stopper = threading.Thread(target=lambda: (arm.stop(), stopped.set()))
    try:
        callback.start()
        assert inspected.wait(1.0)
        stopper.start()
        assert not stopped.wait(0.02), "STOP waits until acceptance is registered"
        release.set()
        callback.join(1.0)
        stopper.join(1.0)
        assert not callback.is_alive() and not stopper.is_alive()
        assert arm._stopped and cancelled.is_set()
    finally:
        release.set()
        callback.join(1.0)
        if stopper.ident is not None:
            stopper.join(1.0)
        arm.disconnect()


@pytest.mark.parametrize("target", [[0.0] * 6, [0.0] * 4, [float('nan')] * 5])
def test_joint_command_requires_exact_finite_dimension(ros2_stub, target):
    arm = _mk_arm()
    _connect(arm)
    try:
        with pytest.raises(ValueError, match="finite joint targets"):
            arm.send_joint_target(target)
        assert not arm._cmd_pub.published
    finally:
        arm.disconnect()


def test_disconnect_stops_spin_thread(ros2_stub):
    arm = _mk_arm()
    _connect(arm)
    spin = arm._spin_thread
    assert spin is not None and spin.is_alive()
    arm.disconnect()
    assert not spin.is_alive()
    assert _FakeNode.instances[-1].destroyed


# ── Complete FollowJointTrajectory motions ──────────────────────────────

class _FakeTrajectoryGoal:
    def __init__(self):
        self.trajectory = _FakeJointTrajectory()
        self.goal_tolerance = []
        self.goal_time_tolerance = _FakeDuration()


@pytest.fixture
def trajectory_action_stub(gripper_action_stub, monkeypatch):
    sys.modules["control_msgs.action"].FollowJointTrajectory = types.SimpleNamespace(Goal=_FakeTrajectoryGoal)
    messages = types.ModuleType("control_msgs.msg")
    messages.JointTolerance = lambda **kw: types.SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "control_msgs.msg", messages)
    yield


def _trajectory_arm(**kw):
    return _mk_arm(ros_trajectory_action="arm_controller/follow_joint_trajectory", **kw)


def _native_result_future(arm, target, *, status=4, code=0, detail="", delay=0.03):
    future = _FakeGoalHandleFuture()
    started = time.monotonic()

    def result():
        if status == 4 and code == 0:
            _feed_state(_FakeNode.instances[-1], target)
        return types.SimpleNamespace(status=status, result=types.SimpleNamespace(
            error_code=code, error_string=detail))

    future.handle.get_result_async = lambda: types.SimpleNamespace(
        done=lambda: time.monotonic() - started >= delay, result=result)
    return future


def _use_native_future(arm, monkeypatch, future):
    client = arm._trajectory_client

    def send(goal):
        client.goals.append(goal)
        return future

    monkeypatch.setattr(client, "send_goal_async", send)
    return client


def test_complete_motion_is_one_quintic_goal_after_full_path_approval(trajectory_action_stub, monkeypatch):
    arm = _trajectory_arm()
    _connect(arm)
    target = np.array([0.08, -0.05, 0.04, 0.0, 0.02])
    future = _native_result_future(arm, target)
    client = _use_native_future(arm, monkeypatch, future)
    validated = []

    def approve(previous, following, dt):
        if not client.goals:
            validated.append((previous.copy(), following.copy(), dt))

    try:
        assert arm.stream_to(target, 0.12, approve=approve, settle_tol=0.015, settle_timeout_s=0.7)
        assert len(client.goals) == 1
        assert not arm._cmd_pub.published, "no 20 ms replacement trajectories"
        assert len(validated) == 6
        np.testing.assert_allclose(validated[-1][1], target)
        assert all(step[2] == pytest.approx(0.02) for step in validated)
        goal = client.goals[0]
        assert goal.trajectory.joint_names == _JOINTS
        first, last = goal.trajectory.points
        assert first.positions == [0.0] * 5 and last.positions == target.tolist()
        assert first.velocities == last.velocities == [0.0] * 5
        assert first.accelerations == last.accelerations == [0.0] * 5
        assert first.time_from_start.sec == first.time_from_start.nanosec == 0
        assert last.time_from_start.nanosec == 120_000_000
        assert [tol.name for tol in goal.goal_tolerance] == _JOINTS
        assert all(tol.position == 0.015 for tol in goal.goal_tolerance)
        assert goal.goal_time_tolerance.nanosec == 700_000_000
        assert not arm.motion_status["motion_active"]
    finally:
        arm.disconnect()


def test_complete_motion_preflight_failure_sends_no_goal(trajectory_action_stub):
    from cascade.safety.harness import SafetyViolation

    arm = _trajectory_arm()
    _connect(arm)

    def reject(previous, following, dt):
        if following[0] > 0.07:
            raise SafetyViolation("future waypoint blocked")

    try:
        with pytest.raises(SafetyViolation, match="future waypoint blocked"):
            arm.stream_to(np.full(5, 0.1), 0.12, approve=reject)
        assert not arm._trajectory_client.goals
        assert arm._stopped
        assert arm._cmd_pub.published[-1].points[0].positions == [0.0] * 5
    finally:
        arm.disconnect()


def test_native_safety_callbacks_use_bounded_duration_before_goal_and_guard_final_feedback(
    trajectory_action_stub, monkeypatch,
):
    arm = _trajectory_arm(ros_max_joint_velocity_rad_s=.5,
                          ros_max_joint_acceleration_rad_s2=.5)
    _connect(arm)
    target = np.full(5, .1)
    client = _use_native_future(arm, monkeypatch, _native_result_future(arm, target, delay=0.))
    events, measured, admitted = [], [], []
    send = client.send_goal_async

    def submit(goal):
        events.append('submit')
        return send(goal)

    monkeypatch.setattr(client, 'send_goal_async', submit)

    def preflight(start, duration):
        events.append('preflight')
        assert not client.goals
        np.testing.assert_array_equal(start, np.zeros(5))
        admitted.append(duration)

    def before():
        events.append('before')
        assert not client.goals

    def feedback(state):
        events.append('feedback')
        measured.append(state.q.copy())

    def approve(previous, following, dt):
        if not client.goals:
            events.append('approve')

    try:
        assert arm.stream_to(target, .1, approve=approve, preflight=preflight,
                             before_stream=before, feedback_guard=feedback, bias_compensate=True)
        assert len(client.goals) == len(admitted) == 1
        endpoint = client.goals[0].trajectory.points[-1]
        actual_duration = endpoint.time_from_start.sec + endpoint.time_from_start.nanosec / 1e9
        assert actual_duration == pytest.approx(admitted[0]) and actual_duration > .1
        assert events.index('before') < events.index('preflight') < events.index('approve')
        assert events[events.index('submit') - 1] == 'before'
        assert events[-1] == 'feedback'
        np.testing.assert_array_equal(measured[-1], target)
        assert endpoint.positions == target.tolist(), 'hardware applies no simulator bias offset'
        assert not arm._cmd_pub.published
    finally:
        arm.disconnect()


@pytest.mark.parametrize('gate', ['preflight', 'before_stream', 'feedback_guard'])
def test_native_callback_rejection_sends_no_goal(trajectory_action_stub, gate):
    from cascade.types import SafetyViolation
    arm = _trajectory_arm()
    _connect(arm)

    def reject(*args):
        raise SafetyViolation('callback refused unsafe motion')

    try:
        with pytest.raises(SafetyViolation, match='callback refused'):
            arm.stream_to(np.full(5, .1), .12, **{gate: reject})
        assert not arm._trajectory_client.goals
        assert arm._stopped
    finally:
        arm.disconnect()


@pytest.mark.parametrize('phase', ['running', 'final'])
def test_native_feedback_guard_failure_cancels_and_holds(trajectory_action_stub, monkeypatch, phase):
    from cascade.types import SafetyViolation
    arm = _trajectory_arm()
    _connect(arm)
    target = np.full(5, .1)
    future = _native_result_future(arm, target, delay=1. if phase == 'running' else 0.)
    client = _use_native_future(arm, monkeypatch, future)

    def feedback(state):
        if client.goals and (phase == 'running' or np.array_equal(state.q, target)):
            raise SafetyViolation('held object observation invalid')

    try:
        with pytest.raises(SafetyViolation, match='held object observation invalid'):
            arm.stream_to(target, .12, feedback_guard=feedback)
        assert len(client.goals) == 1 and future.cancelled and arm._stopped
        hold = arm._cmd_pub.published[-1].points[0]
        expected = np.zeros(5) if phase == 'running' else target
        np.testing.assert_array_equal(hold.positions, expected)
        assert hold.velocities == hold.accelerations == [0.] * 5
    finally:
        arm.disconnect()


def test_native_before_stream_stop_reset_cannot_dispatch_old_goal(trajectory_action_stub):
    arm = _trajectory_arm()
    _connect(arm)

    def stop_reset():
        arm.stop()
        arm.resume()

    try:
        with pytest.raises(RuntimeError, match='interrupted by stop'):
            arm.stream_to(np.full(5, .1), .12, before_stream=stop_reset)
        assert not arm._trajectory_client.goals
        assert not arm._stopped, 'old generation cannot undo an explicit resume'
    finally:
        arm.disconnect()


def test_native_route_preflight_cannot_accept_two_milliradians_of_start_drift(trajectory_action_stub):
    arm = _trajectory_arm()
    _connect(arm)

    def preflight(start, duration):
        _feed_state(_FakeNode.instances[-1], np.full(5, .002))

    try:
        with pytest.raises(RuntimeError, match='moved while validating'):
            arm.stream_to(np.full(5, .1), .12, preflight=preflight)
        assert not arm._trajectory_client.goals
        assert arm._stopped
    finally:
        arm.disconnect()


def test_native_safe_arm_planned_route_keeps_single_bounded_controller_goal(trajectory_action_stub, monkeypatch):
    from cascade.safety.harness import SafeArm, SafetyHarness, SafetyLimits
    arm = _trajectory_arm(ros_max_joint_velocity_rad_s=.5,
                          ros_max_joint_acceleration_rad_s2=.5)
    _connect(arm)
    target = np.full(5, .1)
    # Keep the synthetic feedback below the unchanged measured-velocity gate;
    # an instantaneous .1 rad jump correctly fails even after path admission.
    client = _use_native_future(arm, monkeypatch, _native_result_future(arm, target, delay=.25))
    harness = SafetyHarness(SafetyLimits(np.full(3, -1.), np.full(3, 1.)))
    safe = SafeArm(arm, harness)
    try:
        assert safe.move_planned(target, duration_s=.1)
        assert len(client.goals) == 1 and not arm._cmd_pub.published
        assert not harness.violations and not harness._motion_active
    finally:
        arm.disconnect()


def test_complete_motion_runtime_halt_cancels_and_holds(trajectory_action_stub, monkeypatch):
    from cascade.safety.harness import MotionHalted

    arm = _trajectory_arm()
    _connect(arm)
    future = _native_result_future(arm, np.full(5, 0.1), delay=1.0)
    client = _use_native_future(arm, monkeypatch, future)

    def halt(previous, following, dt):
        if client.goals:
            raise MotionHalted("scene changed")

    try:
        with pytest.raises(MotionHalted, match="scene changed"):
            arm.stream_to(np.full(5, 0.1), 0.12, approve=halt)
        assert future.cancelled and not arm._stopped
        hold = arm._cmd_pub.published[-1].points[0]
        assert hold.positions == [0.0] * 5 and hold.velocities == [0.0] * 5
        assert arm.motion_status["motion_error"] == "scene changed"
        redirected = _native_result_future(arm, np.full(5, 0.1), delay=0.0)
        _use_native_future(arm, monkeypatch, redirected)
        assert arm.stream_to(np.full(5, 0.1), 0.12)
        assert len(client.goals) == 2
    finally:
        arm.disconnect()


@pytest.mark.parametrize("code,name", [(-4, "PATH_TOLERANCE_VIOLATED"), (-5, "GOAL_TOLERANCE_VIOLATED")])
def test_complete_motion_propagates_native_abort(trajectory_action_stub, monkeypatch, code, name):
    arm = _trajectory_arm()
    _connect(arm)
    future = _native_result_future(arm, np.full(5, 0.1), status=6, code=code,
                                   detail="controller joint2 error", delay=0.0)
    _use_native_future(arm, monkeypatch, future)
    try:
        with pytest.raises(RuntimeError, match=name + ".*controller joint2 error"):
            arm.stream_to(np.array([0.05, 0.15, 0.1, 0.0, 0.0]), 0.12, settle_tol=0.02)
        assert future.cancelled and arm._stopped
        assert name in arm.motion_status["motion_error"]
        assert f"{_JOINTS[1]}=0.15000 rad (limit 0.02000 rad)" in arm.motion_status["motion_error"]
    finally:
        arm.disconnect()


def test_complete_motion_rejected_goal_fails_and_holds(trajectory_action_stub, monkeypatch):
    arm = _trajectory_arm()
    _connect(arm)
    future = _FakeGoalHandleFuture(accepted=False)
    _use_native_future(arm, monkeypatch, future)
    try:
        with pytest.raises(RuntimeError, match="goal rejected.*controller activation"):
            arm.stream_to(np.full(5, 0.1), 0.12)
        assert arm._stopped and arm._cmd_pub.published
    finally:
        arm.disconnect()


def test_complete_motion_stop_and_resume_during_validation_cannot_dispatch_old_goal(trajectory_action_stub):
    arm = _trajectory_arm()
    _connect(arm)
    interrupted = False

    def stop(previous, following, dt):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            arm.stop()
            arm.resume()

    try:
        with pytest.raises(RuntimeError, match="interrupted by stop"):
            arm.stream_to(np.full(5, 0.1), 0.12, approve=stop)
        assert not arm._trajectory_client.goals
        assert not arm._stopped, "old motion must not stop a resumed generation"
        assert len(arm._cmd_pub.published) == 1
    finally:
        arm.disconnect()


def test_complete_motion_late_acceptance_is_cancelled_after_resume(trajectory_action_stub):
    arm = _trajectory_arm()
    _connect(arm)
    old_generation = arm._motion_generation
    arm.stop()
    arm.resume()
    current = _FakeGoalHandleFuture()
    late = _FakeGoalHandleFuture()
    try:
        arm._motion_active = True
        arm._on_arm_goal(current, arm._motion_generation)
        arm._on_arm_goal(late, old_generation)
        assert late.cancelled
        assert arm._arm_goal is current.handle
    finally:
        arm._motion_active = False
        arm.disconnect()


def test_complete_motion_unavailable_controller_does_not_dispatch(trajectory_action_stub):
    arm = _trajectory_arm()
    _connect(arm)
    arm._trajectory_client.ready = False
    try:
        with pytest.raises(RuntimeError, match="action server unavailable"):
            arm.stream_to(np.full(5, 0.1), 0.12)
        assert not arm._trajectory_client.goals
        assert arm._stopped
    finally:
        arm.disconnect()


def test_complete_motion_refuses_native_fault_before_arm_or_gripper_goal(trajectory_action_stub):
    from cascade.control.franka_state import FrankaStateMonitor

    arm = _trajectory_arm()
    _connect(arm)
    arm._robot_monitor = FrankaStateMonitor()
    arm._robot_monitor.update(types.SimpleNamespace(
        robot_mode=4, current_errors=types.SimpleNamespace(cartesian_reflex=True),
        last_motion_errors=types.SimpleNamespace(cartesian_reflex=True)))
    try:
        with pytest.raises(RuntimeError, match="cartesian_reflex"):
            arm.stream_to(np.full(5, 0.1), 0.12)
        assert not arm._trajectory_client.goals
        assert not arm.motion_status["motion_ready"]
        arm.resume()
        with pytest.raises(RuntimeError, match="cartesian_reflex"):
            arm.set_gripper(0.6)
        assert not arm._grip_pub.published
    finally:
        arm.disconnect()


def test_complete_motion_native_fault_during_execution_cancels_goal(trajectory_action_stub, monkeypatch):
    from cascade.control.franka_state import FrankaStateMonitor

    arm = _trajectory_arm()
    _connect(arm)
    arm._robot_monitor = FrankaStateMonitor()
    arm._robot_monitor.update(types.SimpleNamespace(robot_mode=2))
    future = _native_result_future(arm, np.full(5, 0.1), delay=1.0)
    client = _use_native_future(arm, monkeypatch, future)

    def fault(previous, following, dt):
        if client.goals:
            arm._robot_monitor.update(types.SimpleNamespace(
                robot_mode=4, current_errors=types.SimpleNamespace(cartesian_reflex=True)))

    try:
        with pytest.raises(RuntimeError, match="cartesian_reflex"):
            arm.stream_to(np.full(5, 0.1), 0.12, approve=fault)
        assert future.cancelled and arm._stopped
        assert "cartesian_reflex" in arm.motion_status["current_errors"]
    finally:
        arm.disconnect()


def test_complete_motion_does_not_accept_success_with_wrong_feedback(trajectory_action_stub, monkeypatch):
    arm = _trajectory_arm()
    _connect(arm)
    future = _native_result_future(arm, np.zeros(5), delay=0.0)
    _use_native_future(arm, monkeypatch, future)
    try:
        with pytest.raises(RuntimeError, match="reported success but feedback error"):
            arm.stream_to(np.full(5, 0.1), 0.12)
        assert arm._stopped
    finally:
        arm.disconnect()


def test_complete_motion_stale_feedback_cancels_queued_goal(trajectory_action_stub, monkeypatch):
    arm = _trajectory_arm()
    _connect(arm)
    future = _native_result_future(arm, np.full(5, 0.1), delay=1.0)
    client = _use_native_future(arm, monkeypatch, future)

    def make_stale(previous, following, dt):
        if client.goals:
            arm._last_state_t = time.monotonic() - 2.0

    try:
        with pytest.raises(RuntimeError, match="JointState.*stale"):
            arm.stream_to(np.full(5, 0.1), 0.12, approve=make_stale)
        assert future.cancelled and arm._stopped
    finally:
        arm.disconnect()


class _PendingGoalFuture:
    def __init__(self):
        self.value = None
        self.callbacks = []

    def done(self):
        return self.value is not None

    def result(self):
        return self.value

    def add_done_callback(self, callback):
        self.callbacks.append(callback)

    def complete(self, handle):
        self.value = handle
        for callback in self.callbacks:
            callback(self)


def test_pending_acceptance_after_runtime_stop_is_cancelled(trajectory_action_stub, monkeypatch):
    arm = _trajectory_arm()
    _connect(arm)
    pending = _PendingGoalFuture()
    client = _use_native_future(arm, monkeypatch, pending)
    stopped = False

    def interrupt(previous, following, dt):
        nonlocal stopped
        if client.goals and not stopped:
            stopped = True
            arm.stop()
            arm.resume()

    try:
        with pytest.raises(RuntimeError, match="interrupted by stop"):
            arm.stream_to(np.full(5, 0.1), 0.12, approve=interrupt)
        accepted = _FakeGoalHandleFuture()
        pending.complete(accepted.handle)
        assert accepted.cancelled
        assert not arm._stopped
        assert arm._arm_goal is None
    finally:
        arm.disconnect()


def test_complete_motion_rejects_start_drift_even_within_settle_tolerance(trajectory_action_stub):
    arm = _trajectory_arm()
    _connect(arm)

    def drift(previous, following, dt):
        _feed_state(_FakeNode.instances[-1], np.full(5, 0.01))

    try:
        with pytest.raises(RuntimeError, match="moved while validating"):
            arm.stream_to(np.full(5, 0.1), 0.12, approve=drift, settle_tol=0.03)
        assert not arm._trajectory_client.goals
    finally:
        arm.disconnect()


def test_complete_motion_rejects_overlapping_native_moves(trajectory_action_stub, monkeypatch):
    arm = _trajectory_arm()
    _connect(arm)
    target = np.full(5, 0.1)
    future = _native_result_future(arm, target, delay=0.0)
    client = _use_native_future(arm, monkeypatch, future)
    attempted = False

    def overlap(previous, following, dt):
        nonlocal attempted
        if not attempted:
            attempted = True
            with pytest.raises(RuntimeError, match="already running"):
                arm.stream_to(target, 0.12)

    try:
        assert arm.stream_to(target, 0.12, approve=overlap)
        assert len(client.goals) == 1
    finally:
        arm.disconnect()


def test_motion_readiness_requires_joint_feedback_even_with_fresh_native_state(trajectory_action_stub):
    from cascade.control.franka_state import FrankaStateMonitor

    arm = _trajectory_arm()
    _connect(arm)
    arm._robot_monitor = FrankaStateMonitor()
    arm._robot_monitor.update(types.SimpleNamespace(robot_mode=2))
    arm._last_state_t = time.monotonic() - 2.0
    try:
        status = arm.motion_status
        assert status["robot_mode"] == "Move"
        assert not status["feedback_valid"] and not status["motion_ready"]
        assert status["feedback_age_s"] >= 2.0
        with pytest.raises(RuntimeError, match="JointState.*stale"):
            arm.ensure_motion_ready()
        with pytest.raises(RuntimeError, match="JointState.*stale"):
            arm.set_gripper(0.6)
        assert not arm._grip_pub.published
    finally:
        arm.disconnect()


def test_actual_motion_velocity_uses_measurement_time_after_duplicate_poll(trajectory_action_stub, monkeypatch):
    arm = _trajectory_arm()
    _connect(arm)
    clock = [time.monotonic()]
    base = clock[0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    _feed_state(_FakeNode.instances[-1], np.zeros(5))
    target = np.full(5, 0.06)
    future = _native_result_future(arm, target, delay=0.075)
    _use_native_future(arm, monkeypatch, future)
    original_state = arm.get_state
    delivered = False
    measured_deltas = []

    def state():
        nonlocal delivered
        if not delivered and clock[0] - base >= 0.039:
            _feed_state(_FakeNode.instances[-1], np.full(5, 0.03))
            delivered = True
        return original_state()

    def approve(previous, following, dt):
        assert np.max(np.abs(following - previous)) / dt <= 1.0 + 1e-6
        if np.allclose(previous, 0.0) and np.allclose(following, 0.03):
            measured_deltas.append(dt)

    monkeypatch.setattr(arm, "get_state", state)
    try:
        assert arm.stream_to(target, 0.12, approve=approve)
        assert measured_deltas == pytest.approx([0.04])
    finally:
        arm.disconnect()


def test_ready_native_state_allows_retry_after_gripper_goal_rejection(gripper_action_stub, monkeypatch):
    from cascade.control.franka_state import FrankaStateMonitor

    arm = _mk_fr3_like()
    _connect(arm)
    arm._robot_monitor = FrankaStateMonitor()
    arm._robot_monitor.update(types.SimpleNamespace(robot_mode=2))
    client = arm._grip_client
    rejected = _FakeGoalHandleFuture(accepted=False)
    accepted = _FakeGoalHandleFuture()
    replies = iter((rejected, accepted))
    monkeypatch.setattr(client, "send_goal_async", lambda goal: next(replies))
    try:
        arm.set_gripper(0.04)
        with pytest.raises(RuntimeError, match="Gripper action goal was rejected"):
            arm.get_state()
        arm.ensure_motion_ready()
        arm.set_gripper(0.04)
        assert arm._grip_goal is accepted.handle
        assert arm.get_state().gripper_valid
    finally:
        arm.disconnect()


def test_connect_subscribes_to_configured_native_franka_state(trajectory_action_stub, monkeypatch):
    messages = types.ModuleType("franka_msgs.msg")
    messages.FrankaRobotState = types.SimpleNamespace
    monkeypatch.setitem(sys.modules, "franka_msgs.msg", messages)
    arm = _trajectory_arm(ros_robot_state_topic="franka_robot_state_broadcaster/robot_state")
    _connect(arm)
    native = next(sub for sub in _FakeNode.instances[-1].subs
                  if sub.topic == "/so101/franka_robot_state_broadcaster/robot_state")
    try:
        native.cb(types.SimpleNamespace(robot_mode=2))
        arm.ensure_motion_ready()
        assert arm.motion_status["motion_ready"]
        native.cb(types.SimpleNamespace(robot_mode=4,
                                       current_errors=types.SimpleNamespace(cartesian_reflex=True)))
        with pytest.raises(RuntimeError, match="cartesian_reflex"):
            arm.ensure_motion_ready()
        assert not arm.motion_status["motion_ready"]
        np.testing.assert_allclose(arm.get_state().q, 0.0)
    finally:
        arm.disconnect()


def test_native_success_cannot_bypass_unsafe_terminal_feedback_within_tolerance(trajectory_action_stub, monkeypatch):
    from cascade.types import SafetyViolation

    arm = _trajectory_arm()
    _connect(arm)
    target = np.full(5, 0.05)
    # The terminal callback reports a pose within the 0.02 rad goal
    # tolerance, but beyond the independently checked 0.06 safety bound.
    # Both planned and measured velocities remain below 1 rad/s.
    future = _native_result_future(arm, np.full(5, 0.065), delay=0.12)
    _use_native_future(arm, monkeypatch, future)

    def approve(previous, following, dt):
        if dt > 0:
            assert np.max(np.abs(following - previous)) / dt <= 1.0
        if np.any(following > 0.06):
            raise SafetyViolation("terminal joint position exceeds safety bound")

    try:
        with pytest.raises(SafetyViolation, match="terminal joint position"):
            arm.stream_to(target, 0.12, approve=approve, settle_tol=0.02)
        assert future.cancelled and arm._stopped
        assert arm._cmd_pub.published[-1].points[0].positions == [0.065] * 5
    finally:
        arm.disconnect()


# ── profile completeness ─────────────────────────────────────────────────

def test_so101_ros2_profile_inherits_the_physical_arm():
    prof = load_profile("arms", "so101_ros2").as_dict()
    assert prof["type"] == "ros2"
    # transport only: the physical constants come from so101.yaml
    assert prof["n_joints"] == 5
    assert len(prof["home_q"]) == 5
    assert prof["gripper"]["max_width_m"] == pytest.approx(0.055)
    assert prof["ros_joints"] == _JOINTS
    # and the rig overrides came through (workspace sized for THIS arm)
    cfg = load_demo_config(camera="mock", arm="so101_ros2", llm="mock")
    assert cfg.safety.workspace.max[0] == pytest.approx(0.40)


def test_ros2_generic_template_loads_and_names_the_contract():
    prof = load_profile("arms", "ros2_generic").as_dict()
    assert prof["type"] == "ros2"
    assert prof["ros_joints"] == []  # deliberately empty: must be filled in
    # The flag that lets a numbers-free profile ship in configs/arms/: the
    # completeness tests skip it BECAUSE the factory refuses to build it.
    # Drop either half and the other stops being justified.
    assert prof["template"] is True
    from cascade.config import Cfg
    from cascade.control.arm_base import make_arm

    with pytest.raises(ValueError, match="TEMPLATE"):
        make_arm(Cfg(prof))


@pytest.mark.parametrize('controller_state', ['active', 'inactive', 'missing'])
def test_resume_preflight_checks_controller_without_clearing_stop(trajectory_action_stub, monkeypatch, controller_state):
    from unittest.mock import Mock
    module = types.ModuleType('controller_manager_msgs.srv')
    module.ListControllers = types.SimpleNamespace(Request=lambda: object())
    monkeypatch.setitem(sys.modules, 'controller_manager_msgs', types.ModuleType('controller_manager_msgs'))
    monkeypatch.setitem(sys.modules, 'controller_manager_msgs.srv', module)
    arm = _trajectory_arm()
    _connect(arm)
    arm._stopped = True
    controllers = [] if controller_state == 'missing' else [
        types.SimpleNamespace(name='arm_controller', state=controller_state)]
    future = types.SimpleNamespace(done=lambda: True,
        result=lambda: types.SimpleNamespace(controller=controllers))
    client = Mock()
    client.wait_for_service.return_value = True
    client.call_async.return_value = future
    arm._node.create_client = Mock(return_value=client)
    arm._node.destroy_client = Mock()
    try:
        if controller_state == 'active':
            arm.check_resume_ready()
        else:
            with pytest.raises(RuntimeError, match='Controller.*inactive'):
                arm.check_resume_ready()
        assert arm._stopped
        assert not arm._trajectory_client.goals
        arm._node.destroy_client.assert_called_once_with(client)
    finally:
        arm.disconnect()


@pytest.mark.parametrize('reason', ['native_fault', 'stale_arm', 'stale_gripper'])
def test_resume_preflight_keeps_stop_when_feedback_is_not_ready(trajectory_action_stub, reason):
    from unittest.mock import Mock
    arm = _trajectory_arm()
    _connect(arm)
    arm._stopped = True
    arm._node.create_client = Mock()
    if reason == 'native_fault':
        arm._robot_monitor = Mock()
        arm._robot_monitor.ensure_ready.side_effect = RuntimeError('Franka Reflex')
        expected = 'Franka Reflex'
    elif reason == 'stale_arm':
        arm._last_state_t = time.monotonic() - 10
        for j in arm._joint_times:
            arm._joint_times[j] = arm._last_state_t
        expected = 'stale'
    else:
        arm._last_gripper_t = time.monotonic() - 10
        expected = 'Gripper feedback is stale'
    try:
        with pytest.raises(RuntimeError, match=expected):
            arm.check_resume_ready()
        assert arm._stopped
        arm._node.create_client.assert_not_called()
        assert not arm._trajectory_client.goals
    finally:
        arm.disconnect()


@pytest.mark.parametrize('status,detail,normal_miss', [
    (6, '', True), (6, 'libfranka network failure', False),
    (5, '', False), (6, None, False),
])
def test_native_grasp_miss_is_distinct_from_driver_fault(
    gripper_action_stub, monkeypatch, status, detail, normal_miss,
):
    from cascade.types import GraspFailed
    arm = _mk_fr3_like(ros_gripper_grasp_action='franka_gripper/grasp', ros_gripper_wait_s=.5)
    _connect(arm)
    try:
        msg = _FakeJointState(); msg.name = ['gripper']; msg.position = [.0071]
        arm._on_joint_state(msg)
        def response(goal):
            future = _FakeGoalHandleFuture()
            future.handle.get_result_async = lambda: types.SimpleNamespace(
                done=lambda: True, result=lambda: types.SimpleNamespace(
                    status=status, result=types.SimpleNamespace(success=False, error=detail)))
            return future
        monkeypatch.setattr(arm._grasp_client, 'send_goal_async', response)
        expected = GraspFailed if normal_miss else RuntimeError
        with pytest.raises(expected) as caught:
            arm.set_gripper(0., effort=.7)
        assert arm._stopped is not normal_miss
        if normal_miss:
            assert 'measured 0.0142 m' in str(caught.value)
            arm.ensure_motion_ready()
        elif detail:
            assert detail in str(caught.value)
    finally:
        arm.disconnect()


@pytest.mark.parametrize('interruption', ['stop', 'stale_gripper', 'stale_arm'])
def test_native_grasp_miss_cannot_bypass_interruption(gripper_action_stub, monkeypatch, interruption):
    from cascade.types import GraspFailed
    arm = _mk_fr3_like(ros_gripper_grasp_action='franka_gripper/grasp', ros_gripper_wait_s=.5)
    _connect(arm)
    try:
        msg = _FakeJointState(); msg.name = ['gripper']; msg.position = [.0071]
        arm._on_joint_state(msg)
        def response(goal):
            def result():
                if interruption == 'stop':
                    arm.stop()
                elif interruption == 'stale_gripper':
                    arm._last_gripper_t = time.monotonic() - 10
                else:
                    arm._last_state_t = time.monotonic() - 10
                return types.SimpleNamespace(status=6, result=types.SimpleNamespace(success=False, error=''))
            future = _FakeGoalHandleFuture()
            future.handle.get_result_async = lambda: types.SimpleNamespace(done=lambda: True, result=result)
            return future
        monkeypatch.setattr(arm._grasp_client, 'send_goal_async', response)
        with pytest.raises(RuntimeError) as caught:
            arm.set_gripper(0.)
        assert not isinstance(caught.value, GraspFailed)
        assert arm._stopped
    finally:
        arm.disconnect()


@pytest.mark.parametrize('target,velocity,acceleration,requested', [
    ([.1, 1.7, -.2, .3, -.4], .5, .5, 3.),
    ([.01, -.02, .005, 0., .01], .5, .1, .05),
    ([.2, -.1, .3, -.4, .1], [.3, .2, .4, .1, .5], [.5, .4, .3, .2, .1], .1),
    ([0., 0., 0., 0., 0.], .5, .5, 2.),
    ([.1, .2, .3, .4, .5], .5, .5, 20.),
])
def test_native_quintic_limits_velocity_and_acceleration_before_dispatch(
    trajectory_action_stub, monkeypatch, target, velocity, acceleration, requested,
):
    arm = _trajectory_arm(ros_max_joint_velocity_rad_s=velocity,
                          ros_max_joint_acceleration_rad_s2=acceleration)
    _connect(arm)
    target = np.asarray(target)
    client = _use_native_future(arm, monkeypatch, _native_result_future(arm, target))
    steps = []
    def approve(prev, nxt, dt):
        if not client.goals:
            steps.append((prev, nxt, dt))
    try:
        assert arm.stream_to(target, requested, approve=approve)
        assert len(client.goals) == 1
        endpoint = client.goals[0].trajectory.points[-1]
        duration = endpoint.time_from_start.sec + endpoint.time_from_start.nanosec / 1e9
        assert duration >= requested - 1e-9
        # Sample derivatives of the actual queued interpolation, independently
        # of the duration helper. Both signs and short/long motions are covered.
        u = np.linspace(0., 1., 10001)[:, None]
        v = (30*u**2 - 60*u**3 + 30*u**4) * target / duration
        a = (60*u - 180*u**2 + 120*u**3) * target / duration**2
        assert np.all(np.max(np.abs(v), axis=0) <= np.asarray(velocity) + 1e-8)
        assert np.all(np.max(np.abs(a), axis=0) <= np.asarray(acceleration) + 1e-8)
        assert sum(step[2] for step in steps) == pytest.approx(duration)
        np.testing.assert_allclose(steps[-1][1], target)
        assert not arm._cmd_pub.published
    finally:
        arm.disconnect()


@pytest.mark.parametrize('key', ['ros_max_joint_velocity_rad_s', 'ros_max_joint_acceleration_rad_s2'])
@pytest.mark.parametrize('invalid', [0., -1., float('nan'), float('inf'), [1., 2.], [1., 1., 0., 1., 1.]])
def test_invalid_native_motion_caps_rejected_before_connection(ros2_stub, key, invalid):
    with pytest.raises(ValueError, match=key):
        _trajectory_arm(**{key: invalid})
