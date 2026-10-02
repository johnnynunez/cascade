"""The actual composed config, passive MCP catalog and stdio sensor path."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET

import pytest

from cascade.apps.mcp_server import McpSkillServer
from cascade.apps.robot_runtime import build_robot_runtime, describe_robot
from cascade.config import load_robot_config
from cascade.robotics.embodiment import EmbodimentDescriptor

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("profile", ["fixed_so101_mock", "wheeled_lift_sensors"])
def test_passive_mcp_description_exposes_structure_without_runtime_or_hardware(profile, monkeypatch):
    monkeypatch.delenv("CASCADE_BASE", raising=False)
    monkeypatch.setenv("CASCADE_ROBOT", profile)
    server = McpSkillServer()
    def forbidden(*args, **kwargs):
        raise AssertionError("passive catalog created a runtime")
    monkeypatch.setattr(server, "_ensure_runtime", forbidden)
    tools = {tool["name"] for tool in server.list_tools()}
    reply = json.loads(server.call_tool("list_resources", {})["content"][-1]["text"])
    body = EmbodimentDescriptor.from_dict(reply["embodiment"])
    assert reply["ok"] and reply["embodiment_sha256"] == body.sha256
    assert not reply["embodiment_provides_transforms"]
    assert all(resource["synthetic"] and resource["admission"] == "software_only" for resource in reply["resources"])
    assert server._runtime is None
    if profile == "wheeled_lift_sensors":
        assert not any("walk" in tool or "pick" in tool or "joint" in tool for tool in tools)
        assert not body.transmissions and body.root_mode == "floating"
    else:
        assert "manipulation.pick_and_place" in tools and body.root_mode == "fixed"


def test_fixed_arm_model_graph_matches_vendored_joint_types_and_units():
    body = EmbodimentDescriptor.from_dict(load_robot_config("fixed_so101_mock").embodiment.as_dict())
    xml = ET.parse(REPO / "assets/urdf/so101/so101.urdf").getroot()
    assert set(body.links) == {link.get("name") for link in xml.findall("link")}
    by_name = {j["joint_id"]: j for j in body.joints}
    for actual in xml.findall("joint"):
        declared = by_name[actual.get("name")]
        assert declared["type"] == actual.get("type")
        assert declared["parent"] == actual.find("parent").get("link")
        assert declared["child"] == actual.find("child").get("link")
        if declared["type"] != "fixed":
            assert declared["axis"] == tuple(float(v) for v in actual.find("axis").get("xyz").split())
            assert dict(declared["limits"]) == {k: float(v) for k, v in actual.find("limit").attrib.items()}
    assert by_name["gripper"]["type"] == "revolute"  # hinged jaw must not become a fictitious linear slide


def test_runtime_catalog_and_sensor_share_same_declaration(tmp_path):
    cfg = load_robot_config("wheeled_lift_sensors")
    rt, owner = build_robot_runtime(cfg, tmp_path)
    try:
        catalog = rt.execute("list_resources")
        observation = rt.execute("sensing.read_sensor", {"sensor_id": "joints"})
        assert observation["ok"], observation
        packet = observation["observation"]
        assert packet["payload"]["embodiment_sha256"] == catalog["embodiment_sha256"]
        assert packet["measurement_kind"] == "synthetic" and packet["model_identity_sha256"] is None
        assert packet["payload"]["joints"][2]["position_unit"] == "m"
        # Returned catalog copies cannot mutate future bindings.
        catalog["embodiment"]["links"].append("forged")
        assert "forged" not in rt.execute("list_resources")["embodiment"]["links"]
    finally:
        assert owner.close()["ok"]


def test_floating_physical_arm_rejected_before_any_builder_opens_io():
    cfg = load_robot_config("fixed_so101_mock")
    cfg._data["embodiment"]["root_mode"] = "floating"
    # Configuration only: never instantiate this physical driver.
    cfg._data["domains"]["manipulation"]["resolved"]["arms"][0]["type"] = "so101"
    with pytest.raises(ValueError, match="dynamic frames"):
        describe_robot(cfg)


def test_existing_mixed_physical_veto_survives_embodiment_work():
    cfg = load_robot_config("mixed_mock")
    cfg._data["domains"]["manipulation"]["resolved"]["arms"][0]["type"] = "so101"
    with pytest.raises(ValueError, match="mixed physical actuation"):
        describe_robot(cfg)


def test_real_stdio_reads_mixed_coordinates_without_fictitious_actuators(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASCADE_")}
    env.update(CASCADE_ROBOT="wheeled_lift_sensors", CASCADE_PREWARM="0", CASCADE_STREAM="0",
        CASCADE_VIEW="0", CASCADE_RUN_DIR=str(tmp_path / "run"),
        CASCADE_BELIEFS_PATH=str(tmp_path / "beliefs.json"),
        CASCADE_GRASP_MEMORY_PATH=str(tmp_path / "grasp.json"), CASCADE_ENVELOPE_PATH=str(tmp_path / "envelope.json"),
        PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="-1")
    proc = subprocess.Popen([sys.executable, "-m", "cascade.apps.mcp_server"], cwd=REPO,
        env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    responses, stderr = queue.Queue(), []
    def read():
        for line in proc.stdout:
            responses.put(json.loads(line))
        responses.put(None)
    threads = [threading.Thread(target=read, daemon=True),
               threading.Thread(target=lambda: stderr.extend(proc.stderr), daemon=True)]
    for thread in threads:
        thread.start()
    def call(index, name, args=None):
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": index, "method": "tools/call",
            "params": {"name": name, "arguments": args or {}}}) + "\n")
        proc.stdin.flush()
        response = responses.get(timeout=15)
        assert response is not None, "".join(stderr)
        assert response["id"] == index and "error" not in response, response
        return json.loads(response["result"]["content"][-1]["text"])
    try:
        catalog = call(1, "list_resources")
        sensor = call(2, "sensing.read_sensor", {"sensor_id": "joints"})
        assert sensor["ok"], sensor
        payload = sensor["observation"]["payload"]
        assert payload["embodiment_sha256"] == catalog["embodiment_sha256"]
        assert [j["position_unit"] for j in payload["joints"]] == ["rad", "rad", "m"]
        assert call(3, "emergency_stop")["latched"]
        assert call(4, "reset_stop")["ok"]
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        finally:
            for thread in threads:
                thread.join(2)
            proc.stdout.close()
            proc.stderr.close()
    assert proc.returncode == 0, "".join(stderr)
