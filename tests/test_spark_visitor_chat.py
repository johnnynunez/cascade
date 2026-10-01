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
    assert second.status()["ready"] is True and second.status()["busy"] is True
    rejection_started = time.monotonic()
    with pytest.raises(chat_module.BusyError):
        second.submit("Move the cube")
    assert time.monotonic() - rejection_started < 1
    release.set()
    result = finished(chat)
    assert result["id"] == submitted["id"] and result["status"] == "done"
    assert chat.status()["ready"] is True
    command, kwargs = commands[0]
    assert command[1:4] == ["--profile", "cascade-demo", "agent"]
    assert command[command.index("--session-id") + 1] == "cascade-proof-" + "a" * 32
    assert kwargs["env"]["OPENCLAW_STATE_DIR"] == str(chat.state / "openclaw")
    assert kwargs["pass_fds"] and "private-token-fixture" not in str(command)


def test_concurrent_health_snapshot_does_not_reject_an_idle_order(chat, monkeypatch):
    observer = chat_module.AttendeeChat(chat.repo)
    snapshot_entered, release_snapshot = threading.Event(), threading.Event()
    submitted, results, errors = threading.Event(), [], []
    actual = observer.unfinished
    def paused_snapshot(session):
        snapshot_entered.set()
        assert release_snapshot.wait(3)
        return actual(session)
    monkeypatch.setattr(observer, "unfinished", paused_snapshot)
    monkeypatch.setattr(chat_module.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=0, stdout=json.dumps(answer(chat))))
    reader = threading.Thread(target=observer.status)
    def submit():
        try:
            results.append(chat.submit("Describe the scene"))
        except Exception as error:
            errors.append(error)
        finally:
            submitted.set()
    writer = threading.Thread(target=submit)
    reader.start()
    try:
        assert snapshot_entered.wait(2)
        writer.start()
        # Only short bookkeeping is held; no worker exists to justify409.
        assert not submitted.wait(.1)
    finally:
        release_snapshot.set()
        reader.join(3)
        if writer.ident is not None:
            writer.join(3)
    assert submitted.is_set() and not errors
    assert len(results) == 1 and finished(chat)["status"] == "done"


def test_concurrent_health_readers_cannot_hide_an_abandoned_order(chat, monkeypatch):
    session = chat.configuration()[0]["session_id"]
    chat.save_order(session, {"id": "c" * 32, "status": "running"})
    observer = chat_module.AttendeeChat(chat.repo)
    entered, release, second_done = threading.Event(), threading.Event(), threading.Event()
    actual = observer.unfinished
    states = []
    def paused_snapshot(session):
        entered.set()
        assert release.wait(3)
        return actual(session)
    monkeypatch.setattr(observer, "unfinished", paused_snapshot)
    first = threading.Thread(target=lambda: states.append(observer.status()))
    def second_read():
        states.append(chat.status())
        second_done.set()
    second = threading.Thread(target=second_read)
    first.start()
    try:
        assert entered.wait(2)
        second.start()
        assert not second_done.wait(.1)
    finally:
        release.set()
        first.join(3)
        if second.ident is not None:
            second.join(3)
    assert len(states) == 2
    assert all(state["ready"] is False and state["busy"] is False for state in states)
    with pytest.raises(chat_module.BusyError):
        chat.submit("Move the cube")


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
    assert chat.status()["ready"] is False
    assert json.loads(chat.result_path.read_text())["order"]["status"] == "uncertain"
    with pytest.raises(chat_module.BusyError):
        chat.submit("Move another cube")
    chat.save_order(chat.configuration()[0]["session_id"], {"id": "a" * 32, "status": "running"})
    assert chat.status()["order"]["status"] == "error"
    assert "private-token-fixture" not in chat.result_path.read_text()


def test_abandoned_order_blocks_both_visitors_until_a_new_verified_world(chat, monkeypatch):
    proof, _ = chat.configuration()
    identifier = "c" * 32
    chat.save_order(proof["session_id"], {"id": identifier, "status": "running"})
    second = chat_module.AttendeeChat(chat.repo)
    assert second.busy() is False
    state = second.status()
    assert state["ready"] is False and state["order"]["status"] == "error"
    assert "cascade-proof" not in json.dumps(state)
    assert second.status("d" * 32)["ready"] is False
    with pytest.raises(chat_module.BusyError):
        second.submit("Please move the orange")
    assert json.loads(chat.result_path.read_text())["order"]["status"] == "running"

    proof["session_id"] = "cascade-proof-" + "b" * 32
    (chat.state / "proof.json").write_text(json.dumps(proof))
    recovered = second.status()
    assert recovered["ready"] is True and recovered["order"]["status"] == "error"
    assert "cascade-proof" not in json.dumps(recovered)
    monkeypatch.setattr(chat_module.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(answer(chat))))
    second.submit("Describe the scene")
    assert finished(second)["status"] == "done"


def test_cli_timeout_keeps_the_uncertain_order_latched(chat, monkeypatch):
    def timeout(command, **kwargs):
        assert command[command.index("--timeout") + 1] == "300"
        assert kwargs["timeout"] == 330
        raise chat_module.subprocess.TimeoutExpired(command, kwargs["timeout"], output="private-token-fixture")
    monkeypatch.setattr(chat_module.subprocess, "run", timeout)
    chat.submit("Move the cube")
    assert finished(chat)["status"] == "error"
    assert chat.status()["ready"] is False
    assert "private-token-fixture" not in chat.result_path.read_text()
    with pytest.raises(chat_module.BusyError):
        chat.submit("Move another cube")


@pytest.mark.parametrize("elapsed_s", [249, 299, 300, 301])
def test_turn_budget_keeps_legitimate_motion_but_expiry_still_latches_stop(chat, monkeypatch, elapsed_s):
    """Model only gateway time; use the real MCP cancel and safety latch.

    No sleeping, physical backend, or claim of an end-to-end OpenClaw test.
    The existing stdio tests separately cover notifications/cancelled routing.
    """
    import numpy as np
    from cascade.apps.mcp_server import McpSkillServer
    from cascade.safety.harness import SafeArm, SafetyHarness, SafetyLimits

    stopped = []
    harness = SafetyHarness(SafetyLimits(np.full(3, -1.), np.full(3, 1.)))
    server = McpSkillServer()
    server._runtime = SimpleNamespace(arm=SafeArm(SimpleNamespace(stop=lambda: stopped.append(True)), harness))
    server._inflight = (41, "pick_and_place")
    calls = []

    def gateway(command, **kwargs):
        calls.append(command)
        deadline = int(command[command.index("--timeout") + 1])
        assert deadline == 300
        assert kwargs["timeout"] == deadline + 30
        envelope = answer(chat)
        envelope["result"]["meta"]["toolSummary"]["tools"] = ["cascade__pick_and_place"]
        if elapsed_s >= deadline:
            # OpenClaw sends this cancellation when its whole-turn budget ends.
            server.cancel_request(41)
            envelope["result"]["meta"]["aborted"] = True
        return SimpleNamespace(returncode=0, stdout=json.dumps(envelope))

    monkeypatch.setattr(chat_module.subprocess, "run", gateway)
    chat.submit("Please put the green cube in the green square.")
    result = finished(chat)
    assert len(calls) == 1
    if elapsed_s < 300:
        assert result["status"] == "done"
        assert not harness.estopped and not server._stop_pending and stopped == []
    else:
        assert result["status"] == "error" and not chat.status()["ready"]
        assert json.loads(chat.result_path.read_text())["order"]["status"] == "uncertain"
        assert harness.estopped and server._stop_pending and stopped == [True]
        with pytest.raises(chat_module.BusyError):
            chat.submit("Move another cube")


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
