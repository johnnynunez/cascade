"""Attendee chat never accepts gateway operations or overlaps robot orders."""
import base64
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

HERE = Path(__file__).resolve().parents[1] / "deploy/brev"


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chat_module = load("visitor_chat")
visitor = load("visitor")


@pytest.fixture
def chat(tmp_path, monkeypatch):
    chat = chat_module.AttendeeChat(tmp_path)
    (chat.state / "openclaw").mkdir()
    proof = {"verified": True, "sim": "isaac", "model": "local/Qwen/Qwen3.8-27B",
             "session_id": "cascade-proof-" + "a" * 32}
    config = {"tools": {"allow": ["cascade__" + name for name in chat_module.TOOLS]},
              "agents": {"defaults": {"skills": []}}, "gateway": {"auth": {"token": "private-token-fixture"}}}
    (chat.state / "proof.json").write_text(json.dumps(proof))
    (chat.state / "openclaw/openclaw.json").write_text(json.dumps(config))
    monkeypatch.setattr(chat_module.AttendeeChat, "live_demo", lambda self, proof: True)
    return chat


def answer(chat, text="Done"):
    proof, _ = chat.configuration()
    return {"status": "ok", "result": {"payloads": [{"text": text}], "meta": {
        "agentMeta": {"sessionId": proof["session_id"], "provider": "local", "model": "Qwen/Qwen3.8-27B"},
        "toolSummary": {"tools": ["cascade__reset_scene"], "calls": 1, "failures": 0}}}}


def finished(chat):
    deadline = time.monotonic() + 3
    while chat.busy() and time.monotonic() < deadline:
        time.sleep(.01)
    assert not chat.busy()
    return chat.status()["order"]


def test_private_profile_identity_and_exact_six_tools_are_required(chat):
    chat.configuration()
    path = chat.state / "openclaw/openclaw.json"
    config = json.loads(path.read_text())
    config["tools"]["allow"].append("exec")
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        chat.submit("Let's start over.")
    assert not chat.result_path.exists()


def test_stale_proof_never_starts_an_embedded_agent(chat, monkeypatch):
    monkeypatch.setattr(chat, "live_demo", lambda proof: False)
    with pytest.raises(ValueError):
        chat.submit("Move the cube")
    assert not chat.result_path.exists()


def test_local_and_public_visitors_share_one_order_lock_and_proof_session(chat, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    commands = []
    def invoke(command, **kwargs):
        commands.append((command, kwargs))
        entered.set()
        assert release.wait(3)
        return SimpleNamespace(returncode=0, stdout=json.dumps(answer(chat)))
    monkeypatch.setattr(chat_module.subprocess, "run", invoke)
    submitted = chat.submit("Let's start over.")
    assert entered.wait(2)
    second = chat_module.AttendeeChat(chat.repo)
    with pytest.raises(chat_module.BusyError):
        second.submit("Move the cube")
    release.set()
    result = finished(chat)
    assert result["id"] == submitted["id"] and result["status"] == "done"
    command, kwargs = commands[0]
    assert command[1:4] == ["--profile", "cascade-demo", "agent"]
    assert command[command.index("--session-id") + 1] == "cascade-proof-" + "a" * 32
    assert kwargs["env"]["OPENCLAW_STATE_DIR"] == str(chat.state / "openclaw")
    assert kwargs["pass_fds"] and "private-token-fixture" not in str(command)


def test_only_assistant_text_and_allowed_tool_names_reach_the_visitor(chat, monkeypatch):
    envelope = answer(chat, "Never publish private-token-fixture")
    envelope["private_token"] = "private-token-fixture"
    monkeypatch.setattr(chat_module.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(envelope)))
    chat.submit("Describe the scene")
    result = finished(chat)
    assert result["message"] == "Never publish [private]"
    assert result["tools"] == ["reset_scene"]
    assert "private-token-fixture" not in chat.result_path.read_text()
    assert "agentMeta" not in chat.result_path.read_text()


def test_cli_failure_and_interrupted_turn_never_claim_success(chat, monkeypatch):
    monkeypatch.setattr(chat_module.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stdout="private-token-fixture"))
    chat.submit("Move the cube")
    assert finished(chat)["status"] == "error"
    chat.save_order(chat.configuration()[0]["session_id"], {"id": "a" * 32, "status": "running"})
    assert chat.status()["order"]["status"] == "error"
    assert "private-token-fixture" not in chat.result_path.read_text()


@contextmanager
def serving(chat):
    server = visitor.VisitorServer(("127.0.0.1", 0), visitor.VisitorHandler)
    server.authorization = "Basic " + base64.b64encode(b"visitor:fixture-password").decode()
    server.chat = chat
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", server.authorization
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)


def post(url, body, authorization=None, **extra):
    headers = {"Content-Type": "application/json", **extra}
    if authorization:
        headers["Authorization"] = authorization
    request = Request(url, data=json.dumps(body).encode(), headers=headers)
    try:
        response = urlopen(request, timeout=3)
    except HTTPError as error:
        response = error
    with response:
        return response.status, response.read()


def test_auth_origin_and_message_allowlist_precede_any_order(chat, monkeypatch):
    monkeypatch.setattr(chat, "submit", lambda message: pytest.fail("Rejected request reached OpenClaw"))
    with serving(chat) as (origin, auth):
        assert post(origin + "/api/chat", {"message": "Move"})[0] == 401
        assert post(origin + "/api/chat", {"message": "Move"}, auth, Origin="https://untrusted.invalid")[0] == 403
        for body in ({"method": "config.set", "params": {}}, {"message": "Move", "agent": "personal"}, {"message": "Move", "session": "other"}):
            assert post(origin + "/api/chat", body, auth)[0] == 400
        assert post(origin + "/api/control", {"message": "Move"}, auth)[0] == 405


def test_authenticated_chat_accepts_only_a_natural_english_message(chat, monkeypatch):
    messages = []
    monkeypatch.setattr(chat, "submit", lambda message: messages.append(message) or {"id": "fixture", "status": "running"})
    with serving(chat) as (origin, auth):
        code, body = post(origin + "/api/chat", {"message": "Please put the orange in the open box."}, auth, Origin=origin)
        assert code == 202 and json.loads(body)["id"] == "fixture"
    assert messages == ["Please put the orange in the open box."]


def test_completed_request_is_available_after_another_visitor_submits(chat, monkeypatch):
    monkeypatch.setattr(chat_module.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(answer(chat))))
    first = chat.submit("Describe the scene")
    finished(chat)
    second = chat.submit("Show the side camera")
    finished(chat)
    assert first["id"] != second["id"]
    assert chat.status(first["id"])["order"]["status"] == "done"
    assert chat.status("../../openclaw/openclaw").get("order") is None


def test_new_proof_does_not_display_an_old_world_result(chat, monkeypatch):
    monkeypatch.setattr(chat_module.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(answer(chat))))
    chat.submit("Describe the scene")
    finished(chat)
    path = chat.state / "proof.json"
    proof = json.loads(path.read_text())
    proof["session_id"] = "cascade-proof-" + "b" * 32
    path.write_text(json.dumps(proof))
    assert chat.status().get("order") is None
