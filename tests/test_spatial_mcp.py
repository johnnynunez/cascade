"""Spatial tools through the real JSON-RPC stdio server, no simulator required."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

from cascade.apps.mcp_server import McpSkillServer


def test_spatial_catalog_does_not_construct_actuators(monkeypatch):
    monkeypatch.delenv("CASCADE_BASE", raising=False)
    monkeypatch.setenv("CASCADE_ROBOT", "spatial_replay")
    server = McpSkillServer()
    def forbidden(*args, **kwargs):
        raise AssertionError("discovery opened a runtime")
    monkeypatch.setattr(server, "_ensure_runtime", forbidden)
    names = {s["name"] for s in server.list_tools()}
    assert {"spatial.get_map", "spatial.recall", "spatial.plan_route", "spatial.lookup_transform"} <= names
    assert not any("walk" in name or "grasp" in name for name in names)
    result = json.loads(server.call_tool("list_resources", {})["content"][-1]["text"])
    assert result["ok"] and len(result["resources"]) == 3
    assert server._runtime is None


def test_stdio_map_memory_plan_and_relocalization_veto(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASCADE_")}
    env.update(CASCADE_ROBOT="spatial_replay", CASCADE_PREWARM="0", CASCADE_STREAM="0",
               CASCADE_VIEW="0", CASCADE_RUN_DIR=str(tmp_path / "run"),
               CASCADE_BELIEFS_PATH=str(tmp_path / "beliefs.json"),
               CASCADE_GRASP_MEMORY_PATH=str(tmp_path / "grasp.json"),
               CASCADE_ENVELOPE_PATH=str(tmp_path / "envelope.json"), PYTHONPATH=str(repo / "src"))
    proc = subprocess.Popen([sys.executable, "-m", "cascade.apps.mcp_server"], cwd=repo, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)
    replies, errors = queue.Queue(), []
    def read():
        for line in proc.stdout:
            replies.put(line)
        replies.put(None)
    threads = [threading.Thread(target=read, daemon=True),
               threading.Thread(target=lambda: errors.extend(proc.stderr), daemon=True)]
    for thread in threads:
        thread.start()
    ident = 0
    def call(name, args=None):
        nonlocal ident
        ident += 1
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": ident, "method": "tools/call",
                                    "params": {"name": name, "arguments": args or {}}})+"\n")
        proc.stdin.flush()
        line = replies.get(timeout=10)
        assert line is not None, errors
        value = json.loads(line)
        assert value["id"] == ident and "error" not in value, value
        return json.loads(value["result"]["content"][-1]["text"])
    try:
        grid = call("spatial.get_map")
        context = dict(time_s=2., epoch="episode-1", clock_id="replay-seconds")
        remembered = call("spatial.recall", {**context, "label": "cup"})
        assert remembered["ok"] and len(remembered["result"]) == 3
        transform = call("spatial.lookup_transform", {**context, "target": "world/map", "source": "spatial_replay/camera"})
        assert transform["ok"] and transform["result"]["matrix"][0][3] == 1.5
        args = {**context, "expected_map_sha256": grid["map_sha256"], "start_xy_m": [.45, .45],
                "goal_xy_m": [2.65, 1.45], "footprint_radius_m": .1, "clearance_m": .02, "position_error_m": .01}
        route = call("spatial.plan_route", args)
        assert route["ok"] and route["result"]["execution"] == "not_executed"
        assert not call("spatial.plan_route", {**args, "epoch": "relocalized"})["ok"]
        assert call("emergency_stop")["latched"]
        assert call("reset_stop")["ok"]
        assert call("spatial.get_map")["map_sha256"] == grid["map_sha256"]
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()
            raise
        finally:
            for thread in threads:
                thread.join(2)
            proc.stdout.close(); proc.stderr.close()
    assert proc.returncode == 0, errors
    trace = [json.loads(line) for line in (tmp_path / "run/trace.jsonl").read_text().splitlines()]
    assert any(row["skill"] == "spatial.plan_route" for row in trace)
