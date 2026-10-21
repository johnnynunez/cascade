"""Humanoid G1 gate hardening (post-merge safety check of #427): CPU only, fake bridges.

1. the agent can never clear a latched stop: ``reset_stop`` is not on a composed robot's agent
   tool list and the runtime refuses it without ``operator=True`` (operator channels keep it);
2. every commanded channel is checked or refused (torso orientation, whole-body-controller leg
   targets, unknown action channels; the bridge refuses an action of another width);
3. a control step without an action is refused (no state-only pass);
4. a malformed message mid-segment latches the stop and the bridge is answered (no hang);
5. the navigation limit is asymmetric per the training data (no backward walking);
6. the agent-episode script defaults to the launch default brain.

Review fix round: the MCP and fleet MCP servers never offer or accept ``reset_stop`` for a robot behind
the humanoid gate (the operator clears it with SIGUSR1 or a restart); refusals never name a tool; a
mid-segment session close keeps the stop latched until the domain is rebuilt; a non-numeric value or a
NaN never passes; ``scripts/replay_gate_jsonl.py`` replays a recorded run's actions offline.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import socket
import threading
from pathlib import Path

import pytest

from cascade.safety.humanoid_gate import HumanoidGateLimits, HumanoidSafetyGate, JointModel
from cascade.skills import humanoid_locomanip as hl

from test_humanoid_locomanip import (IN_BIN, JOINT_MODEL, LIMITS, FakeBridge, action, make_gate, profile, state,
                                     walk_script)

REPO = Path(__file__).resolve().parents[1]
CARRY = {"object": "brown_box", "destination": "blue_sorting_bin"}


def shipped_gate():
    from cascade.config import load_robot_config

    safety = load_robot_config("g1_arena_locomanip").as_dict()["domains"]["carry"]["safety"]
    return HumanoidSafetyGate(HumanoidGateLimits.from_profile(safety), JointModel.from_dict(JOINT_MODEL))


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


def _mcp_server(fake, tmp_path, monkeypatch, robot="g1_arena_locomanip"):
    from cascade.apps.mcp_server import McpSkillServer
    from cascade.skills import humanoid_bridge

    monkeypatch.delenv("CASCADE_BASE", raising=False)
    monkeypatch.delenv("CASCADE_HIDE_TOOLS", raising=False)
    monkeypatch.setenv("CASCADE_ROBOT", robot)
    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("CASCADE_STREAM", "0")
    if fake is not None:
        monkeypatch.setattr(hl, "connect_first_free", lambda host, ports, **kw: humanoid_bridge.BridgeClient(
            host, fake.port, **{k: v for k, v in kw.items() if k == "reply_timeout_s"}))
    return McpSkillServer()


def _payload(result):
    return json.loads(result["content"][-1]["text"])


@pytest.mark.parametrize("args", [{}, {"domain": "carry"}])
def test_the_mcp_agent_can_never_clear_a_humanoid_stop(tmp_path, monkeypatch, args):
    """The MCP host IS the agent: on a robot behind the humanoid gate the MCP server neither lists nor
    accepts reset_stop. The operator's channel (SIGUSR1 -> reset_now, a shell on the rig) still clears it."""
    from cascade.robotics.runtime import OPERATOR_ONLY_ERROR

    fake = FakeBridge(walk_script(2), final=state(99, box_pos=IN_BIN))
    server = _mcp_server(fake, tmp_path, monkeypatch)
    try:
        assert "reset_stop" not in {t["name"] for t in server.list_tools()}
        call = lambda name, a: _payload(server.call_tool(name, a))  # noqa: E731
        assert call("carry.describe_scene", {})["ok"]
        call("emergency_stop", {})
        gate = server._runtime.domains["carry"].runtime.gate
        assert gate.latched is not None
        refused = server.call_tool("reset_stop", args)
        assert refused["isError"] and {k: _payload(refused).get(k) for k in ("ok", "latched", "error")} == {
            "ok": False, "latched": True, "error": OPERATOR_ONLY_ERROR}
        assert gate.latched is not None and server._runtime.stopped and server._stop_pending  # still latched
        carry = call("carry.carry_object", CARRY)
        assert carry["ok"] is False and fake.segments == []
        assert "ask the operator to clear the stop" in carry["error"] and "reset_stop" not in carry["error"]
        # the operator: SIGUSR1 (the staff channel; no MCP tool reaches it) -> reset_now
        assert _payload(server.reset_now())["ok"] is True
        assert gate.latched is None and not server._runtime.stopped
        assert call("carry.carry_object", CARRY)["outcome"] == "confirmed"
    finally:
        receipt = server.shutdown()  # closes the server's runtime (the bridge session included)
    assert receipt["ok"] and receipt["complete"], receipt


@pytest.mark.parametrize("robot", ["fixed_so101_mock", "wheeled_lift_sensors"])
def test_composed_robots_without_the_humanoid_gate_keep_the_mcp_reset_stop(tmp_path, monkeypatch, robot):
    server = _mcp_server(None, tmp_path, monkeypatch, robot)
    assert "reset_stop" in {t["name"] for t in server.list_tools()}
    out = _payload(server.call_tool("reset_stop", {}))  # routed to reset_now: no runtime yet
    assert out["ok"] is False and out["error"] == "runtime starting or input closed; reset refused"


def test_the_arm_mcp_catalog_keeps_reset_stop(monkeypatch):
    from cascade.apps.mcp_server import McpSkillServer

    for key in ("CASCADE_ROBOT", "CASCADE_BASE", "CASCADE_HIDE_TOOLS"):
        monkeypatch.delenv(key, raising=False)
    assert "reset_stop" in {t["name"] for t in McpSkillServer().list_tools()}


def _fleet(tmp_path, *runtimes):
    from cascade.apps.fleet_mcp import FleetMcpServer
    from cascade.robotics.fleet import FleetRuntime
    import queue

    output = queue.Queue()
    server = FleetMcpServer(FleetRuntime(dict(runtimes)), output.put, max_workers=2)
    rpc = lambda ident, name, args=None: {"jsonrpc": "2.0", "id": ident, "method": name if name.startswith("tools/") else  # noqa: E731
                                          "tools/call", "params": {} if name.startswith("tools/") else
                                          {"name": name, "arguments": {} if args is None else args}}
    return server, output, rpc


def test_the_fleet_mcp_agent_can_never_clear_a_humanoid_stop(tmp_path):
    fake = FakeBridge(walk_script(2))
    humanoid = composed_runtime(fake, tmp_path / "g1")
    robot = humanoid.cfg.robot_id
    server, output, rpc = _fleet(tmp_path, (robot, humanoid))
    try:
        server.accept(rpc(1, "tools/list"))
        assert "fleet.reset_stop" not in {t["name"] for t in output.get(timeout=5)["result"]["tools"]}
        server.accept(rpc(2, "fleet.stop", {"robot_id": robot}))
        assert json.loads(output.get(timeout=5)["result"]["content"][-1]["text"])["ok"]
        server.accept(rpc(3, "fleet.reset_stop", {"robot_id": robot}))
        out = json.loads(output.get(timeout=5)["result"]["content"][-1]["text"])
        assert out["ok"] is False and "operator action, not an agent tool" in out["error"]
        assert humanoid.stopped and humanoid.domains["carry"].runtime.gate.latched is not None
    finally:
        assert server.close()["complete"]


def test_a_mixed_fleet_keeps_fleet_reset_stop_for_its_other_robots(tmp_path):
    from test_fleet_runtime import member

    fake = FakeBridge(walk_script(2))
    humanoid = composed_runtime(fake, tmp_path / "g1")
    duck, _ = member("duck", tmp_path)
    robot = humanoid.cfg.robot_id
    server, output, rpc = _fleet(tmp_path, (robot, humanoid), ("duck", duck))
    try:
        server.accept(rpc(1, "tools/list"))
        assert "fleet.reset_stop" in {t["name"] for t in output.get(timeout=5)["result"]["tools"]}
        server.accept(rpc(2, "fleet.stop"))
        assert json.loads(output.get(timeout=5)["result"]["content"][-1]["text"])["ok"]
        server.accept(rpc(3, "fleet.reset_stop", {"robot_id": robot}))
        refused = json.loads(output.get(timeout=5)["result"]["content"][-1]["text"])
        assert refused["ok"] is False and "operator action, not an agent tool" in refused["error"]
        server.accept(rpc(4, "fleet.reset_stop", {"robot_id": "duck"}))
        assert json.loads(output.get(timeout=5)["result"]["content"][-1]["text"])["ok"] is True
        assert humanoid.stopped and not duck.stopped
    finally:
        assert server.close()["complete"]


def test_composed_refusals_never_name_a_tool_the_agent_cannot_call(tmp_path):
    fake = FakeBridge(walk_script(2))
    runtime = composed_runtime(fake, tmp_path)
    try:
        runtime.execute("emergency_stop", {})
        out = runtime.execute("carry.carry_object", CARRY)
        assert out == {"ok": False, "error": "ValueError: robot stopped: ask the operator to clear the stop"}
    finally:
        runtime.close()


@pytest.mark.parametrize("fault", ["garbled", "local"])
def test_after_a_mid_segment_session_close_the_stop_stays_latched_until_rebuilt(tmp_path, monkeypatch, fault):
    fake = ScriptedBridge(walk_script(5), (lambda i, s: "this is not json" if i == 2 else s) if fault == "garbled"
                          else (lambda i, s: s), final=state(99, box_pos=IN_BIN))
    runtime = composed_runtime(fake, tmp_path)
    domain = runtime.domains["carry"].runtime
    if fault == "local":
        real = domain.gate.check

        def check(step, st, act):
            if step == 2:
                raise RuntimeError("disk full")
            return real(step, st, act)

        monkeypatch.setattr(domain.gate, "check", check)
        original = fake._segment

        def segment(msg, f, send):
            try:
                original(msg, f, send)
            except Exception:  # noqa: BLE001 -- the fake reads the close as its gate reply
                pass

        fake._segment = segment
    try:
        runtime.execute("carry.carry_object", CARRY)
        assert domain.gate.latched is not None and fake.applied == 2
        reset = domain.reset_stop()
        assert reset["ok"] is False and "restart" in reset["error"], reset
        out = runtime.execute("reset_stop", {}, operator=True)  # the operator's reset is refused too
        assert out["ok"] is False and runtime.stopped and domain.gate.latched is not None
        carry = runtime.execute("carry.carry_object", CARRY)
        assert carry["ok"] is False and len(fake.segments) == 1 and fake.applied == 2
    finally:
        runtime.close()


def test_operator_scripts_pass_the_operator_flag_to_reset_stop():
    """Every direct ``<runtime>.execute("reset_stop", ...)`` in scripts/ and the evidence harnesses is an
    operator call and passes ``operator=True`` (a composed runtime refuses it otherwise)."""
    import ast

    calls = []
    for path in sorted([*(REPO / "scripts").rglob("*.py"), *(REPO / "docs" / "evidence").rglob("*.py")]):
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "execute"
                    and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "reset_stop"):
                operator = {k.arg: k.value for k in node.keywords}.get("operator")
                calls.append((f"{path.relative_to(REPO)}:{node.lineno}",
                              isinstance(operator, ast.Constant) and operator.value is True))
    assert len(calls) >= 4, calls
    assert [where for where, ok in calls if not ok] == []


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


# ------------------------------------------- 2. every commanded channel is checked or refused
@pytest.mark.parametrize("torso, ok", [
    ([3.0, -3.0, 3.0], False),            # 3 rad: passed the gate before
    ([0.53, 0.0, 0.0], False),            # roll past the waist range (0.52)
    ([0.0, -0.53, 0.0], False),
    ([0.0, 0.0, 2.62], False),            # yaw past the waist yaw range (2.618)
    ([0.5, -0.5, 2.6], True),
    ([0.0, 0.0, 0.0], True),
])
def test_torso_orientation_command_is_limited_from_the_profile(torso, ok):
    act = action(nav=(0.2, 0.0, 0.0))
    act["torso_rpy"] = torso
    d = shipped_gate().check(1, state(1), act)
    assert d.passed is ok
    if not ok:
        assert d.reason.startswith("command_torso_rpy") and [r.name for r in d.rows if not r.ok] == ["command_torso_rpy"]


def test_knee_target_far_out_of_range_stops():
    d = shipped_gate().check(1, state(1), action(targets=[0.0, 25.0, 0.0, 0.0, 0.0]))  # 25 rad knee
    assert d.verdict == "STOP" and d.reason.startswith("command_wbc_joint_rad") and "left_knee_joint" in d.reason


@pytest.mark.parametrize("mutate, word", [
    (lambda a: a.update(extra=[1e9]), "'extra'"),           # a channel the gate does not check
    (lambda a: a.update(mode="walk"), "'mode'"),            # even a non-numeric one
    (lambda a: a.pop("torso_rpy"), "torso_rpy"),            # a channel missing
])
def test_action_channels_the_gate_does_not_check_are_refused(mutate, word):
    act = action()
    mutate(act)
    d = make_gate().check(1, state(1), act)
    assert d.verdict == "STOP" and "malformed step" in d.reason and word in d.reason


NON_NUMBERS = ["nan", "NaN", "inf", "0.1", None, True, [0.1], {"v": 0.1}]


@pytest.mark.parametrize("value", NON_NUMBERS, ids=repr)
@pytest.mark.parametrize("channel, index", [("joint_targets", 2), ("joint_targets", 1), ("joint_targets", 4),
                                            ("nav", 0), ("torso_rpy", 2), ("base_height", None)],
                         ids=["arm", "knee_wbc", "finger", "nav", "torso", "base_height"])
def test_a_non_numeric_command_value_fails_closed(channel, index, value):
    act = action()
    if index is None:
        act[channel] = value
    else:
        act[channel][index] = value
    gate = make_gate()
    d = gate.check(1, state(1), act)
    assert d.verdict == "STOP" and "malformed step" in d.reason and d.clamps == () and gate.latched is not None
    # which check fired: a non-numeric leaf (_flat, strict) or a structure where a number belongs (_num)
    assert ("is not a number" if isinstance(value, (list, dict)) else "non-numeric command value") in d.reason


@pytest.mark.parametrize("value", ["nan", "0.1", None, True], ids=repr)
@pytest.mark.parametrize("key, index", [("joint_pos", 2), ("joint_vel", 1), ("pelvis_pos", 2), ("pelvis_quat_wxyz", 0),
                                        ("floor_z", None)])
def test_a_non_numeric_measured_value_fails_closed(key, index, value):
    st = state(1)
    if index is None:
        st[key] = value
    else:
        st[key][index] = value
    for decide in (lambda g: g.check(1, st, action()), lambda g: g.admit(1, st)):
        d = decide(make_gate())
        assert d.verdict == "STOP" and "malformed step" in d.reason


@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "0.1", "", None, True, False], ids=repr)
def test_every_non_numeric_leaf_of_an_action_is_malformed_while_the_state_may_carry_other_fields(value):
    from cascade.safety.humanoid_gate import _flat

    with pytest.raises(ValueError, match="non-numeric command value"):
        _flat({"nav": [0.1, [value]], "base_height": 0.7}, strict=True)
    assert _flat({"label": "box", "x": [value, 0.5]}) == [0.5]  # every state value the gate reads is _num'd


def _with(base, key, index, value):
    out = copy.deepcopy(base)
    if index is None:
        out[key] = value
    else:
        out[key][index] = value
    return out


NAN = float("nan")


@pytest.mark.parametrize("where, key, index", [
    ("action", "joint_targets", 2), ("action", "joint_targets", 1), ("action", "joint_targets", 4),
    ("action", "nav", 1), ("action", "torso_rpy", 2), ("action", "base_height", None),
    ("state", "joint_pos", 2), ("state", "joint_vel", 0), ("state", "pelvis_pos", 0), ("state", "pelvis_pos", 2),
    ("state", "floor_z", None),
])
def test_a_nan_never_passes_a_comparison_even_past_the_finite_check(monkeypatch, where, key, index):
    """Defence in depth: with the ``state_finite`` pre-check blinded, a NaN still never compares as inside a
    limit (``NaN > worst`` is False, so a plain worst-of loop or ``max`` would skip it)."""
    from cascade.safety import humanoid_gate

    monkeypatch.setattr(humanoid_gate, "_flat", lambda data, **kw: [])
    st, act = state(1), action()
    if where == "action":
        act = _with(act, key, index, NAN)
    else:
        st = _with(st, key, index, NAN)
    d = make_gate().check(1, st, act)
    assert d.verdict == "STOP" and d.clamps == (), d


def _bridge_module():
    spec = importlib.util.spec_from_file_location("arena_locomanip_bridge_under_test",
                                                  REPO / "scripts" / "arena_locomanip_bridge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # stdlib only at import: Arena/torch load inside main()
    return module


class _Tensor:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, i):
        return _Tensor(self.rows[i])

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.rows)


@pytest.mark.parametrize("extra, refused", [(0, False), (1, True), (-1, True)])
def test_bridge_refuses_a_policy_action_of_another_width(extra, refused):
    bridge = _bridge_module().Bridge.__new__(_bridge_module().Bridge)
    bridge.joint_names = ["a", "b", "c"]
    width = 3 + 7 + extra
    action = _Tensor([[0.1 * k for k in range(width)]])
    if refused:
        with pytest.raises(RuntimeError, match="refused, not applied"):
            bridge.split(action)
    else:
        out = bridge.split(action)
        assert set(out) == {"joint_targets", "nav", "base_height", "torso_rpy"} and len(out["joint_targets"]) == 3


# --------------------------------------------------- 3. a control step without an action is refused
def test_policy_step_without_an_action_is_refused_by_the_gate():
    d = make_gate().check(1, state(1), None)
    assert d.verdict == "STOP" and "without a proposed action" in d.reason
    assert make_gate().admit(0, state(0)).passed  # pre-motion admission: the measured state alone


class OwnActionBridge(FakeBridge):
    """At step ``k`` the bridge sends no action and, on ``pass``, would apply its own."""

    def __init__(self, script, k, **kw):
        super().__init__(script, **kw)
        self.k, self.own_applied = k, 0

    def _segment(self, msg, f, send):
        for i, (st, act) in enumerate(self.script):
            step = {"type": "step", "i": i, "phase": "policy", "state": st}
            if i != self.k:
                step["action"] = act
            send(step)
            reply = json.loads(f.readline())
            self.verdicts.append(reply["verdict"])
            if reply["verdict"] != "pass":
                break
            self.applied += 1
            self.own_applied += i == self.k
        for k in range(self.rest):
            send({"type": "step", "i": 10_000 + k, "phase": "hold", "state": self.current, "action": action(nav=(0, 0, 0))})
            self.verdicts.append(json.loads(f.readline())["verdict"])
        send({"type": "segment_end", "reason": "gate_stop", "steps": self.applied, "sim_time_s": 1.0,
              "final": self.current, "arena": None, "recording": None})


def test_domain_refuses_a_policy_step_without_an_action(tmp_path):
    fake = OwnActionBridge(walk_script(5), 2)
    domain = hl.build(profile(fake.port), tmp_path, domain_id="carry")
    out = domain.execute("carry_object", CARRY)
    assert out["stopped"] and out["gate"]["first_stop"]["step"] == 2
    assert fake.own_applied == 0 and fake.applied == 2 and fake.verdicts[:3] == ["pass", "pass", "stop"]
    assert domain.gate.latched is not None
    domain.close()


# ----------------------------------- 4. a malformed message mid-segment latches and is answered
class ScriptedBridge(FakeBridge):
    """``bad(i, step) -> step | str | None``: replace step ``i`` (a str is sent as a raw line).
    Records every line received after a raw line (the answer that un-blocks the bridge)."""

    def __init__(self, script, bad, **kw):
        super().__init__(script, **kw)
        self.bad, self.after_raw = bad, []
        self.answered = threading.Event()

    def _segment(self, msg, f, send):
        for i, (st, act) in enumerate(self.script):
            step = self.bad(i, {"type": "step", "i": i, "phase": "policy", "state": st, "action": act})
            if isinstance(step, str):
                send_raw = getattr(self, "_raw")
                send_raw(step)
                line = f.readline()  # waits like the real bridge's conn.recv()
                self.after_raw.append(json.loads(line) if line else None)
                self.answered.set()
                return
            send(step)
            reply = json.loads(f.readline())
            self.verdicts.append(reply["verdict"])
            if reply["verdict"] != "pass":
                break
            self.applied += 1
        for k in range(self.rest):
            send({"type": "step", "i": 10_000 + k, "phase": "hold", "state": self.current, "action": action(nav=(0, 0, 0))})
            self.verdicts.append(json.loads(f.readline())["verdict"])
        send({"type": "segment_end", "reason": "gate_stop", "steps": self.applied, "sim_time_s": 1.0,
              "final": self.current, "arena": None, "recording": None})

    def _serve(self):
        while True:
            conn, _ = self.sock.accept()
            self._raw = lambda line: conn.sendall((line + "\n").encode())  # noqa: E731
            f = conn.makefile("r")
            send = lambda m: conn.sendall((json.dumps(m) + "\n").encode())  # noqa: E731
            send(self.hello())
            try:
                for line in f:
                    msg = json.loads(line)
                    if msg["type"] == "session":
                        send({"type": "ready", "state": self.current})
                    elif msg["type"] == "observe":
                        send({"type": "state", "state": self.current})
                    elif msg["type"] == "segment":
                        self.segments.append(msg)
                        self._segment(msg, f, send)
                    elif msg["type"] == "close":
                        break
            finally:
                conn.close()


def _bad_index(i, step):
    if i == 2:
        step = copy.deepcopy(step)
        step["i"] = "x"
    return step


@pytest.mark.parametrize("bad", [
    _bad_index,
    lambda i, s: {k: v for k, v in s.items() if k != "state"} if i == 2 else s,        # no state
    lambda i, s: dict(s, state=[1, 2, 3]) if i == 2 else s,                             # state not an object
    lambda i, s: dict(s, action=[0.0] * 12) if i == 2 else s,                           # action not an object
])
def test_malformed_step_mid_segment_latches_and_the_bridge_is_answered(tmp_path, bad):
    fake = ScriptedBridge(walk_script(5), bad, final=state(99, box_pos=IN_BIN))
    domain = hl.build(profile(fake.port), tmp_path, domain_id="carry")
    out = domain.execute("carry_object", CARRY)
    assert domain.gate.latched is not None and "malformed step" in domain.gate.latched
    assert out["stopped"] and out["outcome"] == "stopped", out
    # answered: STOP at the malformed step, the hold steps ran, the segment ended
    assert fake.verdicts == ["pass", "pass", "stop", "stop", "stop"] and fake.applied == 2
    again = domain.execute("carry_object", CARRY)
    assert again["refused"] and len(fake.segments) == 1
    domain.close()


@pytest.mark.parametrize("raw", ["this is not json", json.dumps([1, 2, 3]), json.dumps({"type": "observe"})])
def test_garbled_message_mid_segment_latches_and_ends_the_bridge_session(tmp_path, raw):
    fake = ScriptedBridge(walk_script(5), lambda i, s: raw if i == 2 else s)
    domain = hl.build(profile(fake.port), tmp_path, domain_id="carry")
    out = domain.execute("carry_object", CARRY)
    assert out["ok"] is False and "bridge fault" in out["error"]
    assert domain.gate.latched is not None and fake.applied == 2
    # the bridge, blocked on its gate reply, reads a close (never a pass) before this side lets go
    assert fake.answered.wait(5.0), "the bridge was left waiting for a gate reply"
    assert fake.after_raw == [{"type": "close"}]
    domain.close()


def test_local_fault_mid_segment_latches_and_ends_the_bridge_session(tmp_path, monkeypatch):
    fake = ScriptedBridge(walk_script(5), lambda i, s: s)
    domain = hl.build(profile(fake.port), tmp_path, domain_id="carry")
    real = domain.gate.check

    def check(step, st, act):
        if step == 2:
            raise RuntimeError("disk full")
        return real(step, st, act)

    monkeypatch.setattr(domain.gate, "check", check)
    blocked = threading.Event()
    original = fake._segment

    def segment(msg, f, send):
        try:
            original(msg, f, send)
        except Exception:  # noqa: BLE001 -- the fake reads the close as its gate reply
            blocked.set()

    fake._segment = segment
    with pytest.raises(RuntimeError, match="disk full"):
        domain.execute("carry_object", CARRY)
    assert domain.gate.latched is not None and "disk full" in domain.gate.latched
    assert blocked.wait(5.0), "the bridge was left waiting for a gate reply"
    assert fake.applied == 2
    domain.close()


def test_emergency_stop_mid_segment_latches_at_the_next_control_step(tmp_path):
    fake = ScriptedBridge(walk_script(6), lambda i, s: s, final=state(99, box_pos=IN_BIN))
    domain = hl.build(profile(fake.port), tmp_path, domain_id="carry")
    stop_before = fake.bad

    def bad(i, s):
        if i == 3:
            domain.stop()  # the operator's stop from another thread, between two control steps
        return stop_before(i, s)

    fake.bad = bad
    out = domain.execute("carry_object", CARRY)
    assert out["stopped"] and out["gate"]["first_stop"] == {"step": 3, "reason": "emergency stop requested",
                                                            "phase": "policy"}
    assert fake.applied == 3 and fake.verdicts == ["pass", "pass", "pass", "stop", "stop", "stop"]
    domain.close()


# ------------------------------------------- 5. navigation limit asymmetric per the training data
@pytest.mark.parametrize("vx, ok", [(-0.45, False), (-0.06, False), (-0.05, True), (0.0, True), (0.45, True),
                                    (0.46, False)])
def test_shipped_navigation_limit_is_asymmetric_per_the_training_data(vx, ok):
    d = shipped_gate().check(1, state(1), action(nav=(vx, 0.0, 0.0)))
    assert d.passed is ok
    if not ok:
        assert d.reason.startswith("command_nav")


@pytest.mark.parametrize("value", [[[0.0, 0.4], 0.2, 0.45], [[-0.1, 0.0], 0.2, 0.45], [[-0.1], 0.2, 0.45],
                                   [[-0.1, 0.4, 0.5], 0.2, 0.45], [0.45, 0.2], [-0.45, 0.2, 0.45]])
def test_command_ranges_must_hold_zero_strictly_inside(value):
    with pytest.raises(ValueError):
        HumanoidGateLimits.from_profile(dict(LIMITS, max_nav_command=value))
    with pytest.raises(ValueError):
        HumanoidGateLimits.from_profile(dict(LIMITS, torso_rpy_command_rad=value))


def test_limits_round_trip_and_a_profile_without_the_torso_limit_is_refused():
    lim = HumanoidGateLimits.from_profile(dict(LIMITS, max_nav_command=[[-0.05, 0.45], 0.2, [-0.3, 0.45]]))
    assert lim.max_nav_command == ((-0.05, 0.45), (-0.2, 0.2), (-0.3, 0.45))
    assert HumanoidGateLimits.from_profile(lim.as_dict()) == lim
    with pytest.raises(ValueError):
        HumanoidGateLimits.from_profile({k: v for k, v in LIMITS.items() if k != "torso_rpy_command_rad"})


# ------------------------------------------------ 6. the episode script's default brain
def test_agent_episode_defaults_to_the_launch_default_brain(tmp_path, monkeypatch):
    import cascade.config as config

    spec = importlib.util.spec_from_file_location("humanoid_agent_episode_under_test",
                                                  REPO / "scripts" / "humanoid_agent_episode.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seen = {}

    class Stop(Exception):
        pass

    def load(robot, *, llm="mock", config_dir=None):
        seen["llm"] = llm
        raise Stop

    attempts = []

    def refuse(_sock, address):  # socket guard: this test never reaches a live port (the brain on :8080)
        attempts.append(address)
        raise OSError(f"test socket guard: connect to {address!r} refused")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", lambda address, *a, **k: refuse(None, address))
    monkeypatch.setattr(config, "load_robot_config", load)
    with pytest.raises(Stop):
        module.main(["--robot", "g1_arena_locomanip", "--prompt", "x", "--out", str(tmp_path / "out")])
    assert seen["llm"] == "local_nemotron_bolt" and attempts == []
    # the default brain is the LIVE local server; the script says so (owner convenience, kept)
    assert "base_url: http://127.0.0.1:8080/v1" in (REPO / "configs" / "llm" / "local_nemotron_bolt.yaml").read_text()
    assert "127.0.0.1:8080" in (REPO / "scripts" / "humanoid_agent_episode.py").read_text()
    # the brain scripts/launch.sh resolves by default (--brain auto -> Nemotron 3.5 Bolt)
    assert "nemotron-3.5-bolt" in (REPO / "configs" / "llm" / "local_nemotron_bolt.yaml").read_text()
    launch = (REPO / "scripts" / "launch.sh").read_text()
    assert 'BRAIN="auto"' in launch and "Nemotron 3.5 Bolt by default" in launch


# --------------------------------- 7. offline replay of a recorded run's gate.jsonl (before a real run)
def _replay_module():
    spec = importlib.util.spec_from_file_location("replay_gate_jsonl_under_test", REPO / "scripts" / "replay_gate_jsonl.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recorded_segment(tmp_path):
    """A real record (domain + fake bridge): ``session.json`` and ``segments/001/gate.jsonl``."""
    fake = FakeBridge(walk_script(6), final=state(99, box_pos=IN_BIN))
    domain = hl.build(profile(fake.port), tmp_path / "carry", domain_id="carry")
    assert domain.execute("carry_object", CARRY)["outcome"] == "confirmed"
    domain.close()
    return tmp_path / "carry" / "segments" / "001" / "gate.jsonl"


def _doctor(path, edits):
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["phase"] == "admission" and "action" not in lines[0]
    for key, edit in edits.items():
        step, phase = key if isinstance(key, tuple) else (key, "policy")
        edit(next(line for line in lines if line["step"] == step and line["phase"] == phase)["action"])
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def test_replay_reports_the_first_refused_step_per_channel(tmp_path, capsys):
    path = _recorded_segment(tmp_path)
    _doctor(path, {
        # a leg (WBC) target 0.03 rad past its range: inside the MEASURED tolerance (0.05), past the
        # COMMAND tolerance (0.02): refused (fail closed; the edge case the replay exists to find)
        2: lambda a: a["joint_targets"].__setitem__(1, 2.5 + 0.03),
        3: lambda a: a["nav"].__setitem__(0, -0.3),          # backward: past the shipped vx [-0.05, 0.45]
        4: lambda a: a.update(torso_rpy=[0.6, 0.0, 0.0], nav=[-0.3, 0.0, 0.0]),  # waist roll range; backward again
        5: lambda a: a["joint_targets"].__setitem__(2, "nan"),
        # a finger target past its stop after those refusals: clamped (counted), never refused
        (10_000, "hold"): lambda a: a["joint_targets"].__setitem__(4, -2.0),
    })
    module = _replay_module()
    assert module.main([str(path), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    first = {name: row["first"] and row["first"]["step"] for name, row in report["channels"].items()}
    assert first == {"state_finite": 5, "command_joint_rad": None, "command_wbc_joint_rad": 2, "command_nav": 3,
                     "command_base_height_m": None, "command_torso_rpy": 4}
    assert report["channels"]["command_wbc_joint_rad"]["first"]["detail"] == "left_knee_joint"
    assert {name: row["refused_steps"] for name, row in report["channels"].items()} == {
        "state_finite": 1, "command_joint_rad": 0, "command_wbc_joint_rad": 1, "command_nav": 2,
        "command_base_height_m": 0, "command_torso_rpy": 1}
    assert report["actions"] == 8 and report["skipped"] == 1  # 6 policy + 2 hold steps; the admission line
    assert report["finger_clamps"] == {"steps": 1, "targets": 1}  # each step judged on its own (no latch)
    assert report["limits"]["source"] == "robot profile g1_arena_locomanip (domain carry)"


def test_replay_of_a_clean_record_exits_zero_and_recorded_limits_are_selectable(tmp_path, capsys):
    path = _recorded_segment(tmp_path)
    module = _replay_module()
    assert module.main([str(path)]) == 0
    text = capsys.readouterr().out
    assert "8 recorded actions" in text and "no channel refused" in text
    _doctor(path, {3: lambda a: a["nav"].__setitem__(0, -0.3)})
    assert module.main([str(path), "--json"]) == 1  # the shipped profile: backward walking refused
    capsys.readouterr()
    # the limits the run was recorded with (session.json; symmetric nav 0.45 in the fake profile)
    assert module.main([str(path), "--json", "--recorded-limits"]) == 0
    assert json.loads(capsys.readouterr().out)["limits"]["source"].startswith("recorded session")
    assert module.main([str(tmp_path / "missing.jsonl")]) == 2
