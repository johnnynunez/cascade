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


def test_stop_publishes_empty_trajectory_and_suppresses(ros2_stub):
    """Empty JointTrajectory = ros2_control cancel-and-hold; further
    waypoints are dropped until resume() (soft-stop contract)."""
    arm = _mk_arm()
    _connect(arm)
    try:
        node = _FakeNode.instances[-1]
        cmd = next(p for p in node.pubs
                   if p.topic.endswith("joint_trajectory_controller/joint_trajectory"))
        arm.stop()
        msg = cmd.published[-1]
        assert msg.joint_names == [] and msg.points == []
        n = len(cmd.published)
        arm.send_joint_target(np.zeros(5))   # suppressed
        assert len(cmd.published) == n
        arm.resume()
        arm.send_joint_target(np.zeros(5))
        assert len(cmd.published) == n + 1
    finally:
        arm.disconnect()


def test_stale_state_raises(ros2_stub):
    arm = _mk_arm(state_timeout_s=0.05)
    _connect(arm)
    try:
        time.sleep(0.12)
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
    pass


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
    for name, mod in {"rclpy.action": action_mod, "control_msgs": control_msgs,
                      "control_msgs.action": control_msgs_action}.items():
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
    assert prof["ros_joints"] == [f"fr3_joint{i}" for i in range(1, 8)]
    assert prof["ros_gripper"] == "action"
    assert prof["ros_gripper_action"] == "franka_gripper/gripper_action"
    assert prof["ros_gripper_joint"] == "fr3_finger_joint1"
    assert prof["gripper"]["open_pos"] == pytest.approx(0.04)
    assert prof["gripper"]["max_width_m"] == pytest.approx(0.08)
    assert prof["tool_axis_order"] == "third_open_down"
    cfg = load_demo_config(camera="mock", arm="fr3_mock", llm="mock")
    assert cfg.arm.type == "mock" and cfg.safety.workspace.max[0] == pytest.approx(0.80)


def test_disconnect_stops_spin_thread(ros2_stub):
    arm = _mk_arm()
    _connect(arm)
    spin = arm._spin_thread
    assert spin is not None and spin.is_alive()
    arm.disconnect()
    assert not spin.is_alive()
    assert _FakeNode.instances[-1].destroyed


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
