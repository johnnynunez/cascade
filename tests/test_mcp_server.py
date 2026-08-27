"""MCP stdio server protocol tests: a real subprocess, real JSON-RPC frames.

Uses the mock camera/arm profiles (env), so this covers everything Hermes /
Claude Code exercise except physical hardware.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from conftest import REPO, has_pinocchio

pytestmark = pytest.mark.skipif(not has_pinocchio(), reason="pinocchio not available")


class McpClient:
    def __init__(self, tmp_run_dir: str, extra_env: dict | None = None):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO / "src")
        env["CASCADE_CAMERA"] = "mock"
        env["CASCADE_ARM"] = "mock"
        env["CASCADE_RUN_DIR"] = tmp_run_dir
        env["CASCADE_STREAM"] = "0"  # no HTTP port binding inside tests
        env.update(extra_env or {})
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "cascade.apps.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        self._id = 0
        # stdout is pumped through a queue so recv() can time out instead of
        # hanging the whole suite when a response frame is (wrongly) dropped
        self._out_q: queue.Queue = queue.Queue()

        def _pump():
            for line in self.proc.stdout:
                self._out_q.put(line)
            self._out_q.put(None)  # EOF sentinel

        threading.Thread(target=_pump, daemon=True).start()

    def request(self, method: str, params: dict | None = None, timeout=60):
        self.send(method, params)
        resp = self.recv(timeout=timeout)
        assert resp["id"] == self._id
        return resp

    def send(self, method: str, params: dict | None = None) -> int:
        """Write a request frame WITHOUT waiting for the response -- for
        interleaved traffic (out-of-band emergency_stop)."""
        self._id += 1
        frame = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            frame["params"] = params
        self.proc.stdin.write(json.dumps(frame) + "\n")
        self.proc.stdin.flush()
        return self._id

    def recv(self, timeout=60) -> dict:
        try:
            line = self._out_q.get(timeout=timeout)
        except queue.Empty:
            pytest.fail(f"no response within {timeout}s from a live server "
                        f"(pid {self.proc.pid}); dropped/suppressed frame?")
        assert line is not None, f"server died: {self.proc.stderr.read()[-2000:]}"
        return json.loads(line)

    def notify(self, method: str, params: dict | None = None):
        frame: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            frame["params"] = params
        self.proc.stdin.write(json.dumps(frame) + "\n")
        self.proc.stdin.flush()

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)


@pytest.fixture
def client(tmp_path):
    c = McpClient(str(tmp_path / "run"))
    yield c
    c.close()


def _tool_payload(resp):
    result = resp["result"]
    text = next(b["text"] for b in result["content"] if b["type"] == "text")
    return json.loads(text), result.get("isError", False)


def test_initialize_and_tools_list_without_hardware(client):
    resp = client.request("initialize", {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "pytest", "version": "0"},
    })
    result = resp["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert result["serverInfo"]["name"] == "cascade"
    assert "tools" in result["capabilities"]
    client.notify("notifications/initialized")

    resp = client.request("tools/list")
    tools = {t["name"]: t for t in resp["result"]["tools"]}
    for expected in (
        "get_observation", "grasp_object", "place_at", "push_object",
        "recall_memory", "camera_snapshot", "emergency_stop", "reset_stop",
    ):
        assert expected in tools, f"missing tool {expected}"
    assert "task_done" not in tools  # loop-internal, not exposed
    for t in tools.values():
        assert t["inputSchema"]["type"] == "object"
        assert t["description"]


def test_tools_call_observation_and_memory(client):
    client.request("initialize", {"protocolVersion": "2025-06-18"})
    client.notify("notifications/initialized")

    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "get_observation", "arguments": {}})
    )
    assert not is_err
    assert payload["ok"]
    assert any(o["label"] == "red cube" for o in payload["objects_visible"])

    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "recall_memory",
                                      "arguments": {"query": "red cube"}})
    )
    assert not is_err
    assert "object_memory" in payload


def test_camera_snapshot_returns_image_block(client):
    client.request("initialize", {"protocolVersion": "2025-06-18"})
    resp = client.request("tools/call", {"name": "camera_snapshot", "arguments": {}})
    blocks = resp["result"]["content"]
    img = next(b for b in blocks if b["type"] == "image")
    assert img["mimeType"] == "image/jpeg"
    import base64

    raw = base64.b64decode(img["data"])
    assert raw[:2] == b"\xff\xd8"  # JPEG magic


def test_estop_blocks_motion_until_reset(client):
    client.request("initialize", {"protocolVersion": "2025-06-18"})
    client.request("tools/call", {"name": "get_observation", "arguments": {}})

    payload, _ = _tool_payload(
        client.request("tools/call", {"name": "emergency_stop", "arguments": {}})
    )
    assert payload["stopped"]
    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "move_home", "arguments": {}})
    )
    assert is_err and "e-stop" in payload["error"]
    payload, _ = _tool_payload(
        client.request("tools/call", {"name": "reset_stop", "arguments": {}})
    )
    assert payload["stopped"] is False
    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "move_home", "arguments": {}})
    )
    assert not is_err, payload


def test_unknown_tool_and_method(client):
    client.request("initialize", {"protocolVersion": "2025-06-18"})
    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "warp_drive", "arguments": {}})
    )
    assert is_err and "unknown skill" in payload["error"]
    resp = client.request("frobnicate/all")
    assert resp["error"]["code"] == -32601


def test_ping_and_stdout_purity(client):
    """Every stdout line must be a JSON-RPC frame (Hermes chokes otherwise)."""
    r1 = client.request("initialize", {"protocolVersion": "2024-11-05"})
    assert r1["result"]["protocolVersion"] == "2024-11-05"
    r2 = client.request("ping")
    assert r2["result"] == {}
    r3 = client.request("tools/call", {"name": "get_observation", "arguments": {}})
    assert r3["result"]
    # all three lines parsed as JSON already; nothing else was interleaved


def test_hermes_config_upsert():
    sys.path.insert(0, str(REPO / "scripts"))
    from setup_hermes import upsert, yaml_block

    block = yaml_block("l515", "rebot_rs", "/usr/bin/python3")
    # fresh file
    merged = upsert(None, block)
    assert merged.startswith("mcp_servers:")
    assert 'CASCADE_CAMERAS: "l515"' in merged
    # existing config with another server is preserved
    existing = (
        "model: hermes-4\n"
        "mcp_servers:\n"
        "  agenticros:\n"
        '    command: "node"\n'
        '    args: ["/x/index.js"]\n'
    )
    merged = upsert(existing, block)
    assert "model: hermes-4" in merged
    assert "agenticros:" in merged
    assert "cascade:" in merged
    # idempotent replace: camera changes, no duplicate entries
    block2 = yaml_block("mock", "mock", "/usr/bin/python3")
    merged2 = upsert(merged, block2)
    assert merged2.count("cascade:") == 1
    assert 'CASCADE_CAMERAS: "mock"' in merged2 and 'CASCADE_CAMERAS: "l515"' not in merged2


def test_new_livestream_tools_over_jsonrpc(client):
    """world_state / live_view_url / pick_and_place / camera_snapshot(camera=)
    were untested at the protocol level (2026-07-18 review finding)."""
    client.request("initialize", {"protocolVersion": "2025-06-18"})
    client.notify("notifications/initialized")

    tools = {t["name"] for t in client.request("tools/list")["result"]["tools"]}
    for expected in ("pick_and_place", "world_state", "live_view_url",
                     "describe_scene", "handover", "wave", "point_at"):
        assert expected in tools, f"missing tool {expected}"

    # world_state: instant text, includes objects + cameras, arm stays lazy
    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "world_state", "arguments": {}})
    )
    assert not is_err and payload["ok"]
    assert "objects" in payload and "cameras" in payload
    assert payload["arm_connected"] is False  # LazyArm untouched by a look

    # live_view_url with CASCADE_STREAM=0: honest error, not a bogus URL.
    # The dashboard is lazy now (chat is the UI), so this tool OPENS the view
    # on demand -- but CASCADE_STREAM=0 is a hard kill switch that must still
    # refuse rather than bind a port behind the operator's back.
    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "live_view_url", "arguments": {}})
    )
    assert is_err and payload["open"] is False
    assert "disabled" in payload["error"] and "CASCADE_STREAM" in payload["error"]

    # named-camera snapshot
    resp = client.request("tools/call", {"name": "camera_snapshot",
                                         "arguments": {"camera": "mock"}})
    blocks = resp["result"]["content"]
    assert any(b["type"] == "image" for b in blocks)

    # pick_and_place end-to-end over JSON-RPC (mock jaws close on air ->
    # honest structured failure, never a protocol error)
    payload, is_err = _tool_payload(
        client.request("tools/call",
                       {"name": "pick_and_place", "arguments": {"object": "red object"}},
                       timeout=120)
    )
    assert is_err and payload["stage"] == "grasp"
    assert "grasp failed" in payload["error"]


def test_emergency_stop_preempts_running_motion(client):
    """emergency_stop must NOT queue behind a long motion call: the stdin
    reader latches the e-stop the moment the frame arrives, so its response
    lands FIRST and the running motion aborts mid-stream (booth prep
    2026-07-20; was roadmap near-term item 1)."""
    client.request("initialize", {"protocolVersion": "2025-06-18"})
    client.notify("notifications/initialized")
    # force the runtime up so the stop path has a live harness to latch
    client.request("tools/call", {"name": "get_observation", "arguments": {}})

    pick_id = client.send("tools/call", {"name": "pick_and_place",
                                         "arguments": {"object": "red object"}})
    time.sleep(0.5)  # let the worker enter the motion call
    stop_id = client.send("tools/call", {"name": "emergency_stop", "arguments": {}})

    first = client.recv()
    assert first["id"] == stop_id, "emergency_stop answered after the motion finished"
    payload, _ = _tool_payload(first)
    assert payload["stopped"]

    second = client.recv()
    assert second["id"] == pick_id
    payload, is_err = _tool_payload(second)
    assert is_err
    # the latch must break the persistence loop after the aborted attempt,
    # not burn the remaining retry budget -- "(e-stop latched; not
    # retrying)" is the loop's fail-fast signature (skills/runtime.py)
    assert "not retrying" in payload["error"], payload["error"]

    # the latch holds until reset_stop
    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "move_home", "arguments": {}})
    )
    assert is_err and "e-stop" in payload["error"]


def test_client_cancellation_of_motion_freezes_arm(client):
    """notifications/cancelled for an in-flight MOTION call (Esc in the MCP
    host mid-pick) must freeze the arm, not orphan the motion server-side."""
    client.request("initialize", {"protocolVersion": "2025-06-18"})
    client.request("tools/call", {"name": "get_observation", "arguments": {}})

    pick_id = client.send("tools/call", {"name": "pick_and_place",
                                         "arguments": {"object": "red object"}})
    time.sleep(0.5)
    client.notify("notifications/cancelled", {"requestId": pick_id, "reason": "user"})

    resp = client.recv()  # the aborted call still answers; the host discards it
    assert resp["id"] == pick_id
    _, is_err = _tool_payload(resp)
    assert is_err

    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "move_home", "arguments": {}})
    )
    assert is_err and "e-stop" in payload["error"]


def test_mcp_mode_dashboard_chat_is_reflex_only(tmp_path, monkeypatch):
    """Booth roadmap item 6: in MCP mode the dashboard chat runs the tier-1
    reflex grammar with zero LLM (the host-outage fallback), and refuses
    free-form text with an honest narration note."""
    import urllib.request

    monkeypatch.setenv("CASCADE_CAMERA", "mock")
    monkeypatch.setenv("CASCADE_ARM", "mock")
    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("CASCADE_STREAM", "1")
    monkeypatch.setenv("CASCADE_STREAM_PORT", "0")  # ephemeral port
    monkeypatch.setenv("CASCADE_VIEW", "0")
    from cascade.apps.mcp_server import McpSkillServer

    server = McpSkillServer()
    try:
        runtime = server._ensure_runtime()
        assert runtime.stream_server is not None
        base = f"http://127.0.0.1:{runtime.stream_server.port}"

        def post_task(text: str) -> dict:
            req = urllib.request.Request(
                f"{base}/task", data=json.dumps({"task": text}).encode(),
                headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=5).read())

        assert post_task("wave")["accepted"]
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and runtime.last_path != "reflex":
            time.sleep(0.1)
        assert runtime.last_path == "reflex"

        # free-form text: accepted by HTTP, refused by the handler with a
        # narration note (retry while the wave task drains the busy lock)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if post_task("compose a haiku about the arm")["accepted"]:
                break
            time.sleep(0.2)
        deadline = time.monotonic() + 10
        note_seen = False
        while time.monotonic() < deadline and not note_seen:
            note_seen = any("not a routine command" in ev.text
                            for ev in runtime.memory.events())
            time.sleep(0.1)
        assert note_seen, "free-form chat text must produce the honest note"
    finally:
        server.shutdown()


def test_hidden_tools_are_delisted_and_rejected(tmp_path):
    """CASCADE_HIDE_TOOLS removes tools from the surface AND the call path
    (booth sessions hide reset_stop so a model cannot clear a staff e-stop);
    emergency_stop is never hideable -- even when an operator typo lists it."""
    c = McpClient(str(tmp_path / "run"),
                  extra_env={"CASCADE_HIDE_TOOLS": "reset_stop,throw,emergency_stop"})
    try:
        c.request("initialize", {"protocolVersion": "2025-06-18"})
        tools = {t["name"] for t in c.request("tools/list")["result"]["tools"]}
        assert "reset_stop" not in tools and "throw" not in tools
        assert "emergency_stop" in tools  # survives the hide list

        payload, is_err = _tool_payload(
            c.request("tools/call", {"name": "reset_stop", "arguments": {}})
        )
        assert is_err and "disabled by the operator" in payload["error"]

        # and it is not just listed -- it answers, despite the hide list
        payload, is_err = _tool_payload(
            c.request("tools/call", {"name": "emergency_stop", "arguments": {}})
        )
        assert not is_err and payload["stopped"]
    finally:
        c.close()
