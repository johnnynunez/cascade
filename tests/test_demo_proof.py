from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import demo_proof  # noqa: E402


def _pick(t=100.0, **result_changes):
    result = {
        "ok": True, "verified": True,
        "postcondition": {"status": "confirmed", "channel": "physics"},
    }
    result.update(result_changes)
    return {"t": t, "skill": "pick_and_place", "args": {"object": "red cube"}, "result": result}


def _trace(root, rows, name="mcp_123"):
    path = root / name / "trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_pick_proof_requires_current_physics_evidence(tmp_path):
    check = getattr(demo_proof, "validate_pick_trace", None)
    assert callable(check), "Launcher needs a testable current-run physics validator"
    path = _trace(tmp_path, [_pick()])
    assert check(path, 99.0, "red cube") == path
    _trace(tmp_path, [_pick(t=98.0)])
    with pytest.raises(demo_proof.ProofError, match="current"):
        check(path, 99.0, "red cube")


@pytest.mark.parametrize("result", [
    {"ok": False}, {"verified": False},
    {"postcondition": {"status": "confirmed", "channel": "belief"}},
    {"postcondition": {"status": "unverified", "channel": "physics"}},
])
def test_pick_does_not_accept_a_partial_success(tmp_path, result):
    path = _trace(tmp_path, [_pick(**result)])
    with pytest.raises(demo_proof.ProofError):
        demo_proof.validate_pick_trace(path, 99.0, "red cube")


def test_two_live_worlds_are_ambiguous(tmp_path):
    _trace(tmp_path, [_pick()])
    _trace(tmp_path, [_pick()], "mcp_456")
    with pytest.raises(demo_proof.ProofError):
        demo_proof.validate_pick_trace(tmp_path, 99.0, "red cube")


def test_reset_must_be_in_the_proof_world(tmp_path):
    check = getattr(demo_proof, "validate_reset_trace", None)
    assert callable(check), "Reset needs validation against the exact pick trace"
    path = _trace(tmp_path, [_pick()])
    row = {"t": 120, "skill": "reset_scene", "result": {"ok": True, "world": "mujoco", "props_reset": ["red_cube"]}}
    _trace(tmp_path, [row], "mcp_456")
    with pytest.raises(demo_proof.ProofError, match="reset"):
        check(path, 110, "mujoco", "red cube")
    _trace(tmp_path, [_pick(), row])
    assert check(path, 110, "mujoco", "red cube") == ["red_cube"]


@pytest.mark.parametrize("props", [["blue_cube"], ["red"], ["red_cube_extra"], [], [None]])
def test_reset_requires_the_exact_manipulated_prop(tmp_path, props):
    row = {"t": 120, "skill": "reset_scene", "result": {"ok": True, "world": "mujoco", "props_reset": props}}
    path = _trace(tmp_path, [row])
    with pytest.raises(demo_proof.ProofError, match="prop"):
        demo_proof.validate_reset_trace(path, 110, "mujoco", "red cube")


def test_reset_normalizes_object_name_not_a_colour_substring(tmp_path):
    row = {"t": 120, "skill": "reset_scene", "result": {"ok": True, "world": "mujoco", "props_reset": ["blue_cube", "red_cube"]}}
    path = _trace(tmp_path, [row])
    assert demo_proof.validate_reset_trace(path, 110, "mujoco", " Red  Cube ") == ["blue_cube", "red_cube"]


@pytest.fixture
def host_boundary(tmp_path, monkeypatch):
    """Only the host/simulator are doubles; identity, files and runner are real."""
    import subprocess
    import time
    from cascade.apps import process_owner as owners

    monkeypatch.delenv("CASCADE_OPENCLAW_PROFILE", raising=False)
    monkeypatch.delenv("CASCADE_MCP_NAME", raising=False)
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    owner = owners.load_owner(state, repo, "", create=True)
    info = {"repo": repo, "state": state, "owner": owner, "calls": [], "binding": "new", "children": [], "drop_world_state": False}
    original_run = subprocess.run

    def spawn(record_owner=owner, dead=False):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)",
                                  "cascade.apps.mcp_server", "--launch-owner", record_owner["owner"]])
        info["children"].append(child)
        run_dir = repo / "runs" / f"mcp_{child.pid}_fixture"
        record = owners.register_process(record_owner["state_dir"], record_owner, child.pid, "mcp", run_dir=run_dir)
        if dead:
            child.terminate()
            child.wait(timeout=5)
        info["trace"] = run_dir / "trace.jsonl"
        info["record"] = record
        return record

    info["spawn"] = spawn

    def fake_run(cmd, **kwargs):
        # External OpenClaw boundary double; trace fixtures are NOT GPU proof.
        if cmd[0] != "openclaw":
            return original_run(cmd, **kwargs)
        if "status" in cmd:
            return subprocess.CompletedProcess(cmd, 0, json.dumps({"defaultModel": "openai/test"}), "")
        info["calls"].append(cmd)
        message = cmd[cmd.index("--message") + 1]
        tools = []
        text = "OK"
        if "pick_and_place" in message:
            tools = ["cascade__pick_and_place"]
            trace = info.get("trace", repo / "runs/mcp_foreign/trace.jsonl")
            _trace(repo / "runs", [_pick(t=time.time())], trace.parent.name)
        elif "reset_scene" in message:
            tools = ["cascade__reset_scene"] + ([] if info["drop_world_state"] else ["cascade__world_state"])
            trace = info.get("trace", repo / "runs/mcp_foreign/trace.jsonl")
            with trace.open("a") as f:
                f.write(json.dumps({"t": time.time(), "skill": "reset_scene", "result": {"ok": True, "world": "mujoco", "props_reset": ["red_cube"]}}) + "\n")
        elif "world_state" in message:
            tools = ["cascade__world_state"]
            if info["binding"] in ("new", "ambiguous", "dead"):
                spawn(dead=info["binding"] == "dead")
                if info["binding"] == "ambiguous":
                    spawn()
            elif info["binding"] == "foreign":
                foreign = owners.load_owner(tmp_path / "foreign", repo, "foreign", create=True)
                spawn(foreign)
        data = {"status": "ok", "result": {"payloads": [{"text": text}], "meta": {
            "agentMeta": {"provider": "openai", "model": "test", "sessionId": cmd[cmd.index("--session-id") + 1]},
            "toolSummary": {"calls": len(tools), "tools": tools, "failures": 0},
        }}}
        return subprocess.CompletedProcess(cmd, 0, json.dumps(data), "")

    monkeypatch.setattr(demo_proof.subprocess, "run", fake_run)
    try:
        yield info
    finally:
        for child in info["children"]:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)


def test_proof_uses_one_session_and_checks_real_trace(host_boundary):
    h = host_boundary
    result = demo_proof.run_proof(h["repo"], h["state"], "mujoco", True)
    assert result["verified"] is True
    assert result["trace"] == str(h["trace"])
    assert result["props_reset"] == ["red_cube"]
    calls = h["calls"]
    assert len(calls) == 4
    assert len({cmd[cmd.index("--session-id") + 1] for cmd in calls}) == 1
    assert all("exec" not in cmd for cmd in calls)
    assert "world_state" in calls[1][calls[1].index("--message") + 1]
    assert result["process"]["pid"] == h["record"]["pid"]


@pytest.mark.parametrize("binding", ["foreign", "old", "unregistered", "ambiguous", "dead"])
def test_proof_refuses_unbound_world_before_any_motion(host_boundary, binding):
    h = host_boundary
    h["binding"] = binding
    if binding == "old":
        h["spawn"]()
    # A recent unrelated success must never authorize moving/resetting a world.
    import time
    _trace(h["repo"] / "runs", [_pick(t=time.time())], "mcp_unrelated")
    with pytest.raises(demo_proof.ProofError, match="new|owner|live"):
        demo_proof.run_proof(h["repo"], h["state"], "mujoco")
    assert not any("pick_and_place" in c[c.index("--message") + 1] for c in h["calls"])
    assert json.loads((h["state"] / "proof.json").read_text())["verified"] is False


def test_proof_requires_world_state_after_reset_not_just_in_prompt(host_boundary):
    h = host_boundary
    h["drop_world_state"] = True
    with pytest.raises(demo_proof.ProofError, match="world_state"):
        demo_proof.run_proof(h["repo"], h["state"], "mujoco")


def test_missing_tool_summary_is_not_zero_failures():
    check = getattr(demo_proof, "validate_envelope", None)
    assert callable(check), "Missing verification fields must fail closed"
    with pytest.raises(demo_proof.ProofError, match="toolSummary"):
        check({"status": "ok", "result": {"payloads": [{"text": "OK"}], "meta": {}}}, "pick_and_place")


def test_brain_only_turn_can_omit_tool_summary():
    # Observed OpenClaw 2026.9.3 schema: zero-tool turns omit toolSummary;
    # motion turns may NEVER use its absence as an implicit failures=0.
    result = {"payloads": [{"text": "OK"}], "meta": {}}
    assert demo_proof.validate_envelope({"status": "ok", "result": result}) == result


def test_cosmos_native_probe_rejects_xml_in_prose(monkeypatch):
    check = getattr(demo_proof, "probe_native_tools", None)
    assert callable(check), "OpenClaw bypasses Cosmos3EdgeClient; probe native tool_calls"
    def answer(url, payload=None, timeout=5):
        if url.endswith("/models"):
            return {"data": [{"id": "cosmos3-edge"}]}
        assert payload["tools"][0]["function"]["name"] == "cascade_readiness"
        return {"choices": [{"message": {"content": "<tool_call><function=cascade_readiness></function></tool_call>"}}]}
    monkeypatch.setattr(demo_proof, "request_json", answer)
    with pytest.raises(demo_proof.ProofError, match="native"):
        check("http://127.0.0.1:8082/v1", "cosmos3-edge")


def test_failed_attempt_cannot_leave_an_old_verified_receipt(tmp_path, monkeypatch):
    import subprocess

    state = tmp_path / "state"
    state.mkdir()
    receipt = state / "proof.json"
    receipt.write_text('{"verified": true, "session_id": "previous-run"}')
    monkeypatch.setattr(demo_proof.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "no model"))
    with pytest.raises(demo_proof.ProofError):
        demo_proof.run_proof(tmp_path, state, "mujoco")
    current = json.loads(receipt.read_text())
    assert current["verified"] is False
    assert current["session_id"] != "previous-run"


def test_commands_keep_the_selected_openclaw_profile(monkeypatch):
    monkeypatch.setenv("CASCADE_OPENCLAW_PROFILE", "cascade-demo")
    assert demo_proof.oc_command("models", "status") == ["openclaw", "--profile", "cascade-demo", "models", "status"]
    monkeypatch.delenv("CASCADE_OPENCLAW_PROFILE")
    assert demo_proof.oc_command("models", "status") == ["openclaw", "models", "status"]


def test_gateway_health_retries_a_config_reload(monkeypatch):
    import subprocess

    wait = getattr(demo_proof, "wait_gateway", None)
    assert callable(wait), "Config writes can restart the gateway after its port is already open"
    calls = []
    def health(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1 if len(calls) == 1 else 0, json.dumps({"ok": len(calls) > 1}), "")
    monkeypatch.setattr(demo_proof.subprocess, "run", health)
    assert wait(5, retry_delay=0)["ok"] is True
    assert len(calls) == 2


@pytest.mark.parametrize("session,tools", [
    ("another-session", ["cascade__pick_and_place"]),
    (None, ["cascade__pick_and_place"]),
    ("requested-session", ["foreign__pick_and_place"]),
    ("requested-session", ["foreign_cascade__pick_and_place"]),
])
def test_agent_turn_refuses_wrong_session_or_mcp(tmp_path, monkeypatch, session, tools):
    import subprocess

    monkeypatch.delenv("CASCADE_MCP_NAME", raising=False)
    data = {"status": "ok", "result": {"meta": {
        "agentMeta": {"provider": "openai", "model": "test", "sessionId": session},
        "toolSummary": {"calls": 1, "failures": 0, "tools": tools},
    }}}
    monkeypatch.setattr(demo_proof.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, json.dumps(data), ""))
    with pytest.raises(demo_proof.ProofError, match="session|execute"):
        demo_proof.agent_turn("requested-session", "openai/test", "pick", tmp_path / "turn.json", 10, "pick_and_place")


def test_reset_envelope_requires_separate_world_state_from_exact_server(monkeypatch):
    monkeypatch.setenv("CASCADE_MCP_NAME", "robot-a")
    data = {"status": "ok", "result": {"meta": {"toolSummary": {
        "calls": 2, "failures": 0, "tools": ["robot-a__reset_scene", "foreign__world_state"],
    }}}}
    with pytest.raises(demo_proof.ProofError, match="world_state"):
        demo_proof.validate_envelope(data, ("reset_scene", "world_state"))
    data["result"]["meta"]["toolSummary"]["tools"][1] = "robot-a__world_state"
    assert demo_proof.validate_envelope(data, ("reset_scene", "world_state")) == data["result"]


@pytest.fixture
def model_http_boundary():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    state = {"model": "cosmos3-edge", "requests": []}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            state["requests"].append(self.path)
            payload = json.dumps({"data": [{"id": state["model"]}]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(data)
            payload = {"choices": [{"message": {"tool_calls": [{"type": "function", "function": {
                "name": "cascade_readiness", "arguments": '{"ready":true}',
            }}]}}]}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_auto_brain_wrong_model_is_a_hard_error_not_qwen_fallback(model_http_boundary, monkeypatch):
    h = model_http_boundary
    h["model"] = "not-cosmos"
    monkeypatch.setenv("CASCADE_COSMOS_BASE_URL", h["base_url"])
    monkeypatch.setenv("CASCADE_QWEN_BASE_URL", h["base_url"])
    with pytest.raises(demo_proof.ProofError, match="not served"):
        demo_proof.check_brain("auto")
    assert len(h["requests"]) == 1


def test_auto_brain_only_falls_back_on_connection_failure(model_http_boundary, monkeypatch):
    h = model_http_boundary
    h["model"] = "qwen-test"
    monkeypatch.setenv("CASCADE_COSMOS_BASE_URL", "http://127.0.0.1:0/v1")
    monkeypatch.setenv("CASCADE_QWEN_BASE_URL", h["base_url"])
    info = demo_proof.check_brain("auto")
    assert info["brain"] == "qwen"
    assert info["model"] == "qwen-test"
    assert info["base_url"] == h["base_url"]
