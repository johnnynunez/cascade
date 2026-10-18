"""The same profile must describe the robot to planner, MCP and controllers."""
import json
from pathlib import Path

import pytest
import yaml

from base_pose_oracle import bind_like_primary
from cascade.agent.prompts import SYSTEM_PROMPT, build_system_prompt
from cascade.apps.mcp_server import McpSkillServer, handle_message
from cascade.config import load_demo_config
from cascade.skills.runtime import SkillRuntime, TOOL_SPECS, tool_specs_for_config

ROOT = Path(__file__).resolve().parents[1]


def fr3():
    return load_demo_config(arm="fr3", camera="d435i", llm="mock")


def test_fr3_prompt_describes_actual_frames_transport_and_gripper():
    prompt = build_system_prompt(fr3())
    for fact in ["Franka Research 3", "7-DOF", "1..7", "fr3_hand_tcp", "fr3_link0",
                 "Franka Gripper", "0.08 m", "0.04", "franka_bringup", "MoveIt is not used",
                 "eye_in_hand", "camera moves", "FCI"]:
        assert fact in prompt
    assert "operator-taught overview pose" in prompt
    assert "ordinary failed grasp" in prompt
    assert "STOP, hardware faults and stale feedback prevent automatic return" in prompt
    assert "reBot" not in prompt and "6-DOF" not in prompt
    assert "reBot" not in SYSTEM_PROMPT


def test_schemas_follow_profile_without_cross_arm_mutation():
    original = json.dumps(TOOL_SPECS, sort_keys=True)
    a = {t["name"]: t for t in tool_specs_for_config(fr3())}
    b = {t["name"]: t for t in tool_specs_for_config(
        load_demo_config(arm="so101_mock", camera="mock", llm="mock"))}
    assert a["rotate_joint"]["parameters"]["properties"]["joint"]["maximum"] == 7
    assert b["rotate_joint"]["parameters"]["properties"]["joint"]["maximum"] == 5
    assert "throw" not in a and "throw" in b
    assert "Franka Gripper" in a["close_gripper"]["description"]
    assert "0.08 m" in a["grasp_object"]["description"]
    assert "saved operator camera-view home" in a["pick_and_place"]["description"]
    assert "Composite pick/place phases stay together" in a["pick_and_place"]["description"]
    assert "return_home" in a["grasp_object"]["description"]
    assert "camera-view home" not in b["grasp_object"]["description"]
    assert "Franka" not in b["close_gripper"]["description"]
    assert json.dumps(TOOL_SPECS, sort_keys=True) == original


def test_mcp_discovery_reports_fr3_without_materializing_hardware(monkeypatch):
    monkeypatch.setenv("CASCADE_ARM", "fr3")
    monkeypatch.setenv("CASCADE_CAMERA", "d435i")
    monkeypatch.delenv("CASCADE_CAMERAS", raising=False)
    server = McpSkillServer()
    monkeypatch.setattr(server, "_ensure_runtime", lambda: pytest.fail("Discovery opened hardware"))
    init = handle_message(server, {"id": 1, "method": "initialize"})["result"]
    assert "7-DOF" in init["instructions"]
    assert "task_done" not in init["instructions"]
    tools = handle_message(server, {"id": 2, "method": "tools/list"})["result"]["tools"]
    by_name = {t["name"]: t for t in tools}
    assert "throw" not in by_name
    assert by_name["rotate_joint"]["inputSchema"]["properties"]["joint"]["maximum"] == 7
    assert server._runtime is None


def test_unsupported_fr3_throw_rejected_before_observation_or_motion():
    runtime = SkillRuntime.__new__(SkillRuntime)
    from cascade.agent.task_effects import TaskEffects
    runtime._task_effects = TaskEffects()
    runtime.arm_rig = None
    runtime.held_object = None
    runtime.arm = object()
    runtime.cfg = fr3()
    runtime.observe = lambda: pytest.fail("Unsupported skill observed/moved the arm")
    response = runtime.execute("throw", {"label": "box"})
    assert response["ok"] is False and "unsupported" in response["error"]


def test_controller_configuration_matches_fr3_profile():
    cfg = fr3()
    controllers = yaml.safe_load((ROOT / "configs/ros2/fr3_controllers.yaml").read_text())
    manager = controllers["/controller_manager"]["ros__parameters"]
    arm = controllers["/fr3_arm_controller"]["ros__parameters"]
    assert manager["fr3_arm_controller"]["type"] == "joint_trajectory_controller/JointTrajectoryController"
    assert arm["joints"] == cfg.arm.ros_joints
    assert arm["command_interfaces"] == ["effort"]
    assert set(arm["gains"]) == set(cfg.arm.ros_joints)
    assert cfg.arm.gripper.min_effort_n == 20
    assert cfg.arm.gripper.max_effort_n == 70
    assert cfg.arm.ros_gripper_move_action == "franka_gripper/move"
    assert cfg.arm.ros_gripper_grasp_action == "franka_gripper/grasp"
    assert cfg.arm.ros_trajectory_action == "fr3_arm_controller/follow_joint_trajectory"
    assert cfg.arm.ros_robot_state_topic == "franka_robot_state_broadcaster/robot_state"
    assert cfg.arm.return_home_after_motion is True
    assert cfg.grasp.backend == "obb"
    assert "/fr3/" in cfg.memory.experience_path


def test_planner_uses_profile_for_decomposition_and_agent_turns():
    from types import SimpleNamespace
    from cascade.agent.llm import MockLLM, LLMResponse
    from cascade.agent.orchestrator import AgentOrchestrator

    llm = MockLLM([LLMResponse(text="1. Observe the scene")])
    agent = AgentOrchestrator(llm, SimpleNamespace(cfg=fr3()), verify_milestones=False)
    assert agent._decompose("observe") == ["Observe the scene"]
    assert "7-DOF" in llm.requests[0]["system"]
    assert "Franka Gripper" in agent.system_prompt
    assert "throw" not in {s["name"] for s in agent.tool_specs}


def test_faulted_motion_skill_is_refused_before_camera_or_gripper_commands():
    from types import SimpleNamespace

    class FaultedArm:
        def ensure_motion_ready(self):
            raise RuntimeError("Franka hardware not ready: current_errors: cartesian_reflex")

    runtime = SkillRuntime.__new__(SkillRuntime)
    from cascade.agent.task_effects import TaskEffects
    runtime._task_effects = TaskEffects()
    runtime.arm_rig = None
    runtime.held_object = None
    runtime.arm = object()
    runtime.cfg = fr3()
    runtime.arm = SimpleNamespace(raw=FaultedArm())
    runtime._show_status = lambda status: None
    runtime.observe = lambda: pytest.fail("Faulted skill observed the scene")
    runtime.skill_open_gripper = lambda: pytest.fail("Faulted skill opened the gripper")
    result = runtime.execute("open_gripper", {})
    assert not result["ok"]
    assert result["motion_blocked"]
    assert "cartesian_reflex" in result["error"]


def test_motion_readiness_is_checked_on_the_selected_arm():
    from types import SimpleNamespace
    from cascade.config import Cfg

    class FaultedArm:
        def ensure_motion_ready(self):
            raise RuntimeError("selected arm fault")

    runtime = SkillRuntime.__new__(SkillRuntime)
    from cascade.agent.task_effects import TaskEffects
    runtime._task_effects = TaskEffects()
    runtime.arm_rig = None
    runtime.held_object = None
    runtime.arm = object()
    runtime.cfg = fr3()
    primary = runtime.cfg.arm.as_dict()
    runtime.cfg = Cfg({**runtime.cfg.as_dict(),
                       "arms": [primary, {**primary, "name": "second"}]})
    runtime.arm = SimpleNamespace(raw=object())
    selected = SimpleNamespace(raw=FaultedArm())
    runtime.arm_rig = SimpleNamespace(get=lambda name: selected, arms={"second": selected})
    bind_like_primary(runtime, "second")  # execute(arm=) refuses an arm with no binding
    runtime._show_status = lambda status: None
    runtime.observe = lambda: pytest.fail("Selected faulted arm was not gated")
    result = runtime.execute("open_gripper", {"arm": "second"})
    assert not result["ok"] and "selected arm fault" in result["error"]


def _dashboard_runtime(raw):
    from types import SimpleNamespace

    return SimpleNamespace(
        arm=SimpleNamespace(raw=raw, get_state=lambda: object(),
                            harness=SimpleNamespace(estopped=False)),
        camera=object(), current_task=None, held_object=None,
        beliefs=SimpleNamespace(summary=lambda: []),
        memory=SimpleNamespace(digest=lambda **kwargs: ""),
        grasp_memory=SimpleNamespace(summary=lambda: ""),
        backends=lambda: {}, watcher=None,
    )


def test_dashboard_distinguishes_joint_connection_from_motion_readiness():
    from cascade.apps.demo import _runtime_state

    class FaultedArm:
        connected = True

        @property
        def motion_status(self):
            return {"motion_ready": False, "robot_mode": "Reflex",
                    "current_errors": ["cartesian_reflex"]}

    state = _runtime_state(_dashboard_runtime(FaultedArm()))
    assert state["arm_connected"] is True
    assert state["arm_motion"]["motion_ready"] is False
    assert state["arm_motion"]["current_errors"] == ["cartesian_reflex"]


def test_dashboard_readiness_does_not_materialize_a_lazy_arm():
    from cascade.apps.demo import _runtime_state
    from cascade.control.lazy_arm import LazyArm

    arm = LazyArm(lambda: pytest.fail("Dashboard powered hardware"))
    state = _runtime_state(_dashboard_runtime(arm))
    assert state["arm_connected"] is False
    assert "arm_motion" not in state
