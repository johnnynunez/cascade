"""Real stdio composition catalog and independent mock-domain dispatch."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

import pytest

from conftest import needs_pin_so101
from cascade.apps.mcp_server import McpSkillServer


REPO = Path(__file__).resolve().parents[1]


def unpack(response):
    assert "error" not in response, response
    return json.loads(response["result"]["content"][-1]["text"])


class Client:
    def __init__(self, directory):
        env = {k: v for k, v in os.environ.items() if not k.startswith("CASCADE_")}
        env.update(CASCADE_ROBOT="mixed_mock", CASCADE_PREWARM="0", CASCADE_STREAM="0",
                   CASCADE_VIEW="0", CASCADE_RUN_DIR=str(directory),
                   CASCADE_BELIEFS_PATH=str(directory / "beliefs.json"),
                   CASCADE_GRASP_MEMORY_PATH=str(directory / "grasp.json"),
                   CASCADE_ENVELOPE_PATH=str(directory / "envelope.json"),
                   PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="-1")
        self.proc = subprocess.Popen([sys.executable, "-m", "cascade.apps.mcp_server"],
            cwd=REPO, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        self.queue, self.errors, self.next_id = queue.Queue(), [], 0
        self.threads = [threading.Thread(target=self._read, daemon=True),
                        threading.Thread(target=lambda: self.errors.extend(self.proc.stderr), daemon=True)]
        for thread in self.threads:
            thread.start()

    def _read(self):
        for line in self.proc.stdout:
            self.queue.put(line)
        self.queue.put(None)

    def send(self, name, arguments=None):
        self.next_id += 1
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.next_id, "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}}}) + "\n")
        self.proc.stdin.flush()
        return self.next_id

    def receive(self, timeout=15):
        line = self.queue.get(timeout=timeout)
        assert line is not None, "".join(self.errors)
        return json.loads(line)

    def call(self, name, arguments=None):
        ident = self.send(name, arguments)
        result = self.receive()
        assert result["id"] == ident
        return unpack(result)

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
            raise
        finally:
            for thread in self.threads:
                thread.join(2)
            self.proc.stdout.close()
            self.proc.stderr.close()


def test_catalog_is_passive_and_separates_domains(monkeypatch):
    monkeypatch.delenv("CASCADE_BASE", raising=False)
    monkeypatch.setenv("CASCADE_ROBOT", "mixed_mock")
    server = McpSkillServer()
    def forbidden(*args, **kwargs):
        raise AssertionError("catalog opened a runtime")
    monkeypatch.setattr(server, "_ensure_runtime", forbidden)
    tools = {s["name"] for s in server.list_tools()}
    assert {"list_resources", "emergency_stop", "reset_stop", "manipulation.pick_and_place",
            "locomotion.walk_velocity", "sensing.read_sensor"} <= tools
    assert "walk_velocity" not in tools and "pick_and_place" not in tools
    info = json.loads(server.call_tool("list_resources", {})["content"][-1]["text"])
    assert info["ok"] and all(r["synthetic"] for r in info["resources"])
    assert server._runtime is None


@needs_pin_so101
def test_stdio_composes_observation_arm_base_sensor_and_preserves_verdict(tmp_path):
    client = Client(tmp_path / "stdio")
    try:
        assert client.call("list_resources")["ok"]
        sensor = client.call("sensing.read_sensor", {"sensor_id": "imu"})
        assert sensor["ok"], sensor
        observation = client.call("manipulation.get_observation")
        assert observation["ok"] and observation["robot"]["live_arm_feedback"] is False
        assert client.call("manipulation.open_gripper")["ok"]
        observed_after_command = client.call("manipulation.get_observation")
        assert "q_deg" in observed_after_command["robot"]  # explicit mock motion opened the arm
        bases = client.call("locomotion.list_bases")
        assert bases["bases"][0]["measurement_kind"] == "kinematic_mock"
        result = client.call("locomotion.walk_velocity", {"vx": .05, "vy": 0, "wz": 0, "duration_s": .06})
        assert result["execution_ok"] and not result["ok"]
        assert result["postcondition"]["status"] == "unverified"
        stopped = client.call("emergency_stop")
        assert stopped["latched"] and set(stopped["domains"]) == {"manipulation", "locomotion", "sensing"}
        assert client.call("reset_stop")["ok"]
    finally:
        client.close()
    assert client.proc.returncode == 0, "".join(client.errors)


def test_composition_and_base_environment_are_not_ambiguous(monkeypatch):
    monkeypatch.setenv("CASCADE_ROBOT", "mixed_mock")
    monkeypatch.setenv("CASCADE_BASE", "microduck_mock")
    with pytest.raises(ValueError, match="mutually exclusive"):
        McpSkillServer()


def test_composed_cli_signal_requests_graceful_shutdown_before_teardown(tmp_path, monkeypatch):
    import signal
    from types import SimpleNamespace
    from cascade.apps import demo

    monkeypatch.delenv("CASCADE_BASE", raising=False)
    monkeypatch.delenv("CASCADE_ROBOT", raising=False)
    events = []
    def forbidden():
        raise AssertionError("CLI signal manufactured an e-stop before arm parking")
    runtime = SimpleNamespace(
        request_shutdown=lambda: events.append("graceful_shutdown"), stop=forbidden)
    monkeypatch.setattr(demo, "build_runtime", lambda *args, **kwargs: (runtime, runtime))
    def interrupted(*args):
        signal.raise_signal(signal.SIGTERM)
        raise AssertionError("signal did not interrupt the CLI")
    monkeypatch.setattr(demo, "_run_demo", interrupted)
    def teardown(actual_runtime, owner):
        assert actual_runtime is runtime and owner is runtime
        events.append("teardown")
    monkeypatch.setattr(demo, "shutdown_runtime", teardown)
    assert demo.main(["--robot", "mixed_mock", "--llm", "mock", "--no-view", "--no-serve",
                      "--run-dir", str(tmp_path / "cli-signal")]) == 128 + signal.SIGTERM
    assert events == ["graceful_shutdown", "teardown"]


@needs_pin_so101
def test_composed_cli_mock_is_an_explicit_passive_catalog_check(tmp_path, monkeypatch, capsys):
    from cascade.apps.demo import main
    monkeypatch.delenv("CASCADE_BASE", raising=False)
    monkeypatch.delenv("CASCADE_ROBOT", raising=False)
    assert main(["--robot", "mixed_mock", "--llm", "mock", "--no-view", "--no-serve",
                 "--run-dir", str(tmp_path / "cli")]) == 0
    output = capsys.readouterr().out
    assert "inspected declared resources; no physical task executed" in output
    records = [json.loads(row) for row in (tmp_path / "cli" / "trace.jsonl").read_text().splitlines()]
    assert [r["skill"] for r in records] == ["list_resources", "task_done"]


@needs_pin_so101
def test_explicit_offline_composition_never_probes_grasp_or_occupancy(tmp_path, monkeypatch):
    from cascade.apps.robot_runtime import build_robot_runtime
    from cascade.config import load_robot_config
    from cascade.grasping.graspgenx_backend import GraspGenXPlanner
    from cascade.perception.occupancy import OccupancyMap
    monkeypatch.delenv("CASCADE_ROBOT", raising=False)
    cfg = load_robot_config("mixed_mock")
    def forbidden(*args, **kwargs):
        raise AssertionError("offline composition contacted an optional service")
    monkeypatch.setattr(GraspGenXPlanner, "probe", forbidden)
    original = OccupancyMap.from_config
    def checked(config, **kwargs):
        assert config.get("enabled") is False
        return original(config, **kwargs)
    monkeypatch.setattr(OccupancyMap, "from_config", checked)
    rt, _ = build_robot_runtime(cfg, tmp_path)
    try:
        assert rt.execute("manipulation.get_observation", {})["ok"]
        arm = rt.domains["manipulation"].runtime
        assert arm.grasp_planner_used == "obb"
        assert arm.arm.harness.occupancy is None
    finally:
        assert rt.close()["ok"]
