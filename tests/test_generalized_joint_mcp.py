"""Passive discovery and actual stdio observations; never a robot-motion proof."""
import json
import time

from cascade.apps.mcp_server import McpSkillServer
from cascade.apps.robot_runtime import build_robot_runtime
from cascade.config import load_robot_config
from test_robot_mcp import Client


def check_packet(catalog, result):
    assert result["ok"], result
    observation = result["observation"]
    payload = observation["payload"]
    assert payload["modality"] == "generalized_joint_state"
    assert observation["measurement_kind"] == "synthetic"
    assert observation["model_identity_sha256"] is None
    assert payload["embodiment_sha256"] == catalog["embodiment_sha256"]
    assert catalog["embodiment"]["version"] == 2
    assert not catalog["embodiment_provides_transforms"]
    assert not catalog["embodiment"]["transmissions"]
    assert [(len(j["q"]), len(j["v"])) for j in payload["joints"]] == [(3, 3), (4, 3), (7, 6)]
    assert all(j["effort"] is None for j in payload["joints"])
    assert payload["joints"][1]["coordinates"]["rotation"] == "hamilton_wxyz_child_to_parent"
    assert all(r["kind"] == "sensor" and r["writer_id"] is None for r in catalog["resources"])


def test_generalized_discovery_is_passive_and_adds_no_movement_tool(monkeypatch):
    monkeypatch.delenv("CASCADE_BASE", raising=False)
    monkeypatch.setenv("CASCADE_ROBOT", "generalized_joint_sensors")
    server = McpSkillServer()
    def forbidden(*args, **kwargs):
        raise AssertionError("passive discovery constructed a runtime")
    monkeypatch.setattr(server, "_ensure_runtime", forbidden)
    names = {tool["name"] for tool in server.list_tools()}
    assert {"sensing.read_sensor", "sensing.list_sensors", "list_resources"} <= names
    assert not any("walk" in name or "pick" in name or "joint" in name or "turn" in name for name in names)
    catalog = json.loads(server.call_tool("list_resources", {})["content"][-1]["text"])
    assert catalog["ok"] and catalog["embodiment"]["version"] == 2 and server._runtime is None


def test_ordinary_runtime_reads_generalized_payload_without_control(tmp_path):
    runtime, owner = build_robot_runtime(load_robot_config("generalized_joint_sensors", llm="mock"), tmp_path)
    try:
        assert not runtime.motion_skills
        catalog = runtime.execute("list_resources")
        result = runtime.execute("sensing.read_sensor", {"sensor_id": "joints"})
        check_packet(catalog, result)
        result["observation"]["payload"]["joints"][0]["q"][0] = 999
        # SensorHub rejects replay; wait one synthetic capture period, not a new epoch.
        time.sleep(.02)
        assert runtime.execute("sensing.read_sensor", {"sensor_id": "joints"})["observation"]["payload"]["joints"][0]["q"][0] == .1
    finally:
        assert owner.close()["ok"]


def test_real_stdio_generalized_read_and_stop_remain_observation_only(tmp_path):
    client = Client(tmp_path / "stdio", robot="generalized_joint_sensors")
    try:
        catalog = client.call("list_resources")
        check_packet(catalog, client.call("sensing.read_sensor", {"sensor_id": "joints"}))
        assert client.call("emergency_stop")["latched"]
        time.sleep(.02)  # preserve the existing rejection of reads on the same capture
        check_packet(catalog, client.call("sensing.read_sensor", {"sensor_id": "joints"}))
        assert client.call("reset_stop")["ok"]
    finally:
        client.close()
    assert client.proc.returncode == 0, "".join(client.errors)
