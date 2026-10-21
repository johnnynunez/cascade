"""Humanoid G1 gate hardening (post-merge safety check of #427): CPU only, fake bridges.

1. the agent can never clear a latched stop: ``reset_stop`` is not on a composed robot's agent
   tool list and the runtime refuses it without ``operator=True`` (operator channels keep it);
"""
from __future__ import annotations

import json
import os

import pytest

from cascade.skills import humanoid_locomanip as hl

from test_humanoid_locomanip import IN_BIN, FakeBridge, state, walk_script

CARRY = {"object": "brown_box", "destination": "blue_sorting_bin"}


def composed_runtime(fake, tmp_path):
    from cascade.apps.robot_runtime import build_robot_runtime
    from cascade.config import Cfg, load_robot_config

    cfg = load_robot_config("g1_arena_locomanip").as_dict()
    cfg["domains"]["carry"]["bridge"]["ports"] = [fake.port]
    runtime, _ = build_robot_runtime(Cfg(cfg), tmp_path)
    return runtime


# ------------------------------------------------- 1. only the operator clears a latched stop
def test_composed_agent_tool_list_has_no_reset_stop(tmp_path):
    from cascade.agent.llm import MockLLM
    from cascade.agent.orchestrator import AgentOrchestrator

    fake = FakeBridge(walk_script(2))
    runtime = composed_runtime(fake, tmp_path)
    try:
        names = {t["name"] for t in runtime.tool_specs}
        assert "reset_stop" not in names
        assert {"emergency_stop", "task_done", "list_resources", "carry.carry_object"} <= names
        assert "reset_stop" in runtime.tool_descriptors  # the operator path keeps it
        orch = AgentOrchestrator(MockLLM([]), runtime, decompose=False, verify_milestones=False)
        assert "reset_stop" not in {t["name"] for t in orch.tool_specs}
    finally:
        runtime.close()


def test_the_agent_cannot_clear_a_latched_stop(tmp_path):
    from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
    from cascade.agent.orchestrator import AgentOrchestrator

    fake = FakeBridge(walk_script(6, tilt_at=3), final=state(99, box_pos=IN_BIN))
    runtime = composed_runtime(fake, tmp_path)
    try:
        first = runtime.execute("carry.carry_object", CARRY)
        assert first["outcome"] == "stopped" and len(fake.segments) == 1
        domain = runtime.domains["carry"].runtime
        assert domain.gate.latched is not None
        fake.current, fake.script = state(50), walk_script(2)  # a nominal world from here on
        llm = MockLLM([
            LLMResponse("", [ToolCall("reset_stop", {}, "1")]),
            LLMResponse("", [ToolCall("carry.carry_object", CARRY, "2")]),
            LLMResponse("", [ToolCall("task_done", {"success": False, "summary": "stopped"}, "3")]),
        ])
        report = AgentOrchestrator(llm, runtime, decompose=False, verify_milestones=False).run_task(
            "put the brown box in the blue bin")
        reset, carry = report.tool_log[0]["result"], report.tool_log[1]["result"]
        assert reset["ok"] is False
        assert reset["error"] == ("ValueError: reset_stop is an operator action, not an agent tool: "
                                  "ask the operator to clear the stop")
        assert carry["refused"] and carry["error"].startswith("stopped: ask the operator to clear the stop")
        assert len(fake.segments) == 1 and fake.applied == 3  # nothing moved after the agent's reset attempt
        assert domain.gate.latched is not None
    finally:
        runtime.close()


@pytest.mark.parametrize("operator", [False, None, 1, "yes"])
def test_reset_stop_without_the_operator_flag_is_refused(tmp_path, operator):
    fake = FakeBridge(walk_script(2))
    runtime = composed_runtime(fake, tmp_path)
    try:
        runtime.execute("emergency_stop", {})
        out = runtime.execute("reset_stop", {}, operator=operator)
        assert out["ok"] is False and "operator action" in out["error"]
        assert runtime.stopped and runtime.domains["carry"].runtime.gate.latched is not None
    finally:
        runtime.close()


def test_the_operator_clears_the_stop_and_the_robot_runs_again(tmp_path):
    fake = FakeBridge(walk_script(2), final=state(99, box_pos=IN_BIN))
    runtime = composed_runtime(fake, tmp_path)
    try:
        runtime.execute("emergency_stop", {})
        assert runtime.execute("carry.carry_object", CARRY)["ok"] is False and fake.segments == []
        reset = runtime.execute("reset_stop", {}, operator=True)
        assert reset["ok"] is True and not runtime.stopped
        assert runtime.domains["carry"].runtime.gate.latched is None
        assert runtime.execute("carry.carry_object", CARRY)["outcome"] == "confirmed"
    finally:
        runtime.close()


def test_mcp_reset_stop_stays_the_operator_channel_for_composed_robots(tmp_path, monkeypatch):
    from cascade.apps.mcp_server import McpSkillServer
    from cascade.skills import humanoid_bridge

    fake = FakeBridge(walk_script(2), final=state(99, box_pos=IN_BIN))
    monkeypatch.delenv("CASCADE_BASE", raising=False)
    monkeypatch.setenv("CASCADE_ROBOT", "g1_arena_locomanip")
    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(hl, "connect_first_free", lambda host, ports, **kw: humanoid_bridge.BridgeClient(
        host, fake.port, **{k: v for k, v in kw.items() if k == "reply_timeout_s"}))
    server = McpSkillServer()
    assert "reset_stop" in {t["name"] for t in server.list_tools()}  # MCP catalog unchanged
    call = lambda name, args: json.loads(server.call_tool(name, args)["content"][-1]["text"])  # noqa: E731
    assert call("carry.describe_scene", {})["ok"]
    call("emergency_stop", {})
    assert server._runtime.domains["carry"].runtime.gate.latched is not None
    assert call("reset_stop", {})["ok"] is True
    assert server._runtime.domains["carry"].runtime.gate.latched is None
    assert call("carry.carry_object", CARRY)["outcome"] == "confirmed"


def test_operator_console_reset_passes_the_operator_flag_on_composed_runtimes():
    from cascade.apps.mobile_runtime import run_mobile_interactive

    class Recorder:
        def __init__(self, mode):
            self.robot_mode, self.calls = mode, []

        def execute(self, name, args, **kwargs):
            self.calls.append((name, args, kwargs))
            return {"ok": True}

        def stop(self):
            return {"ok": True}

    for mode, expected in (("composed", {"operator": True}), ("mobile", {})):
        runtime = Recorder(mode)
        read_fd, write_fd = os.pipe()
        os.write(write_fd, b"reset_stop\n")
        os.close(write_fd)
        with os.fdopen(read_fd) as stream:
            run_mobile_interactive(runtime, lambda _t: None, lambda _r: None, input_stream=stream)
        assert runtime.calls == [("reset_stop", {}, expected)]


def test_harness2_operator_arm_keeps_its_operator_reset(tmp_path, monkeypatch):
    from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
    from cascade.harnessing import AgentHarness, CascadeExecutor
    from cascade.harnessing.executor import TaskSpec
    from cascade.skills import humanoid_bridge

    fake = FakeBridge(walk_script(2))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hl, "connect_first_free", lambda host, ports, **kw: humanoid_bridge.BridgeClient(
        host, fake.port, **{k: v for k, v in kw.items() if k == "reply_timeout_s"}))
    llm = lambda cfg, label: MockLLM([LLMResponse("", [ToolCall("task_done", {"success": False, "summary": "x"})])])  # noqa: E731
    ex = CascadeExecutor(robot="g1_arena_locomanip", llm_profile="mock", llm_factory=llm, decompose=False,
                         verify_milestones=False, operator_arm=True)
    rollout = ex.run(AgentHarness(), TaskSpec("c1", "look", "g1_carry", max_steps=3), "base1", tmp_path / "r")
    assert rollout.error is None, rollout.error
    assert rollout.ended_by == "task_done"
