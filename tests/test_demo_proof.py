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


def test_generic_configured_drop_point_keeps_the_physics_proof_contract(tmp_path):
    from cascade.agent.effects import PostconditionChecker, annotate_result

    args = {"object": "red cube", "destination": "drop zone"}
    result = {"ok": True, "picked": "red cube", "placed_at": [.30, -.20, .1],
              "destination": "drop zone", "destination_kind": "configured_point"}
    checker = PostconditionChecker(object_pose=lambda name: [.305, -.205, .03])
    verdict = checker.verify("pick_and_place", args, result,
        {"label": "red cube", "pose": [.1, .1, .03], "channel": "physics"})
    result = annotate_result(result, verdict)
    assert result["verified"] is True
    assert verdict.status == "confirmed" and verdict.channel == "physics"
    assert verdict.measured["target_err_m"] < .01
    path = _trace(tmp_path, [{"t": 100.0, "skill": "pick_and_place",
                             "args": args, "result": result}])
    assert demo_proof.validate_pick_trace(path, 99.0, "red cube") == path


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
    import select
    import subprocess
    import time
    from cascade.apps import process_owner as owners

    monkeypatch.delenv("CASCADE_OPENCLAW_PROFILE", raising=False)
    monkeypatch.delenv("CASCADE_MCP_NAME", raising=False)
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    owner = owners.load_owner(state, repo, "", create=True)
    info = {"repo": repo, "state": state, "owner": owner, "calls": [], "binding": "new", "children": [], "drop_world_state": False,
            "reply": "OK", "model": "openai/test"}
    original_run = subprocess.run

    def spawn(record_owner=owner, dead=False):
        child = subprocess.Popen(
            [sys.executable, "-c", "import os, time; os.write(1, b'R'); time.sleep(120)",
             "cascade.apps.mcp_server", "--launch-owner", record_owner["owner"]],
            stdout=subprocess.PIPE,
        )
        info["children"].append(child)
        # Popen can return during exec with /proc/<pid>/cmdline still empty.
        # Wait for the final Python child; never relax real ownership checks.
        assert child.stdout is not None
        assert select.select([child.stdout], [], [], 10)[0], (
            f"fixture child {child.pid} did not become ready (exit={child.poll()})"
        )
        assert child.stdout.read(1) == b"R", (
            f"fixture child {child.pid} exited or sent invalid readiness (exit={child.poll()})"
        )
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
            return subprocess.CompletedProcess(cmd, 0, json.dumps({"defaultModel": info["model"]}), "")
        info["calls"].append(cmd)
        message = cmd[cmd.index("--message") + 1]
        tools = []
        text = info["reply"]
        if "pick_and_place" in message or message.startswith("Please put the"):
            tools = ["cascade__pick_and_place"]
            trace = info.get("trace", repo / "runs/mcp_foreign/trace.jsonl")
            _trace(repo / "runs", [_pick(t=time.time())], trace.parent.name)
        elif "reset_scene" in message or message.startswith("Let's start over"):
            tools = ["cascade__reset_scene"] + ([] if info["drop_world_state"] else ["cascade__world_state"])
            trace = info.get("trace", repo / "runs/mcp_foreign/trace.jsonl")
            with trace.open("a") as f:
                f.write(json.dumps({"t": time.time(), "skill": "reset_scene", "result": {"ok": True, "world": "mujoco", "props_reset": ["red_cube"]}}) + "\n")
        elif "world_state" in message or message == "What is the session status?":
            tools = ["cascade__world_state"]
            if info["binding"] in ("new", "ambiguous", "dead"):
                spawn(dead=info["binding"] == "dead")
                if info["binding"] == "ambiguous":
                    spawn()
            elif info["binding"] == "foreign":
                foreign = owners.load_owner(tmp_path / "foreign", repo, "foreign", create=True)
                spawn(foreign)
        data = {"status": "ok", "result": {"payloads": [{"text": text}], "meta": {
            "agentMeta": {"provider": info["model"].partition("/")[0], "model": info["model"].partition("/")[2],
                          "sessionId": cmd[cmd.index("--session-id") + 1]},
            "toolSummary": {"calls": len(tools), "tools": tools, "failures": 0},
        }}}
        return subprocess.CompletedProcess(cmd, 0, json.dumps(data), "")

    monkeypatch.setattr(demo_proof.subprocess, "run", fake_run)
    try:
        yield info
    finally:
        for child in info["children"]:
            try:
                if child.poll() is None:
                    child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            finally:
                if child.stdout is not None:
                    child.stdout.close()


def test_host_boundary_waits_for_final_exec_before_registration(host_boundary, monkeypatch):
    import select
    import subprocess
    from cascade.apps import process_owner as owners

    original_popen, original_select = subprocess.Popen, select.select
    children, waits = [], []

    def gated_popen(cmd, **kwargs):
        if cmd[:2] != [sys.executable, "-c"]:
            return original_popen(cmd, **kwargs)
        # Hold a real intermediate interpreter until the fixture waits for
        # readiness. Only the final exec may emit the fixture's ready byte.
        trampoline = ("import os, sys\n"
                      "if os.read(0, 1) != b'G': os._exit(72)\n"
                      "os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n")
        child = original_popen([sys.executable, "-c", trampoline, *cmd[1:]],
                               stdin=subprocess.PIPE, **kwargs)
        children.append(child)
        return child

    def release_exec(readers, writers, errors, timeout):
        assert readers == [children[0].stdout]
        waits.append(timeout)
        children[0].stdin.write(b"G")
        children[0].stdin.close()
        return original_select(readers, writers, errors, timeout)

    monkeypatch.setattr(subprocess, "Popen", gated_popen)
    monkeypatch.setattr(select, "select", release_exec)
    try:
        record = host_boundary["spawn"]()
        assert waits == [10], "identity was read before waiting for the final child"
        assert "os.execv" not in record["command"], "registered the intermediate interpreter"
        assert owners.is_live(record, host_boundary["owner"])
        assert owners.records(host_boundary["state"]) == [record]
    finally:
        for child in children:
            child.stdin.close()


@pytest.mark.parametrize("failure", ["timeout", "eof", "invalid", "stubborn", "registration"])
def test_host_boundary_cleans_children_after_spawn_failure(tmp_path, monkeypatch, failure):
    import select
    import subprocess
    from cascade.apps import process_owner as owners

    # Drive the real yield fixture so its finalizer can be checked inside the
    # test, including a child which never acquired an ownership receipt.
    boundary = host_boundary.__wrapped__(tmp_path, monkeypatch)
    h = next(boundary)
    original_popen, original_select = subprocess.Popen, select.select
    original_register = owners.register_process
    created, registrations = [], []
    sources = {
        "timeout": "import os; os.read(0, 1)",
        "eof": "pass",
        "invalid": "import os, time; os.write(1, b'X'); time.sleep(120)",
        "stubborn": ("import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                     "os.write(1, b'X'); time.sleep(120)"),
    }

    def faulty_popen(cmd, **kwargs):
        if cmd[:2] != [sys.executable, "-c"]:
            return original_popen(cmd, **kwargs)
        cmd = list(cmd)
        cmd[2] = sources.get(failure, cmd[2])
        child = original_popen(cmd, stdin=subprocess.PIPE, **kwargs)
        created.append(child)
        return child

    def register(*args, **kwargs):
        registrations.append(args[2])
        if failure == "registration":
            raise ValueError("injected registration failure")
        return original_register(*args, **kwargs)

    try:
        with monkeypatch.context() as faults:
            faults.setattr(subprocess, "Popen", faulty_popen)
            faults.setattr(owners, "register_process", register)
            if failure == "timeout":
                # The child blocks on a real pipe. Exercise select's timeout
                # result immediately, not by sleeping or retrying registration.
                faults.setattr(select, "select", lambda r, w, e, timeout: original_select(r, w, e, 0))
            error = ValueError if failure == "registration" else AssertionError
            message = "injected registration failure" if failure == "registration" else "ready|readiness"
            with pytest.raises(error, match=message):
                h["spawn"]()
            assert len(created) == 1
            assert h["children"] == created, "failed spawn escaped fixture cleanup"
            assert registrations == ([created[0].pid] if failure == "registration" else [])
            assert owners.records(h["state"]) == []
        # A failed/stubborn child must not prevent cleanup of later children.
        record = h["spawn"]()
        assert owners.is_live(record, h["owner"])
        boundary.close()
        assert len(h["children"]) == 2
        assert all(child.returncode is not None for child in h["children"]), "child was not reaped"
        assert all(child.stdout.closed for child in h["children"]), "readiness pipe leaked"
    finally:
        try:
            boundary.close()
        finally:
            # Test-owned stdin and an emergency guard keep a regressed
            # finalizer from leaking real processes during the RED test.
            for child in created + [c for c in h["children"] if c not in created]:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)
                for stream in (child.stdin, child.stdout):
                    if stream is not None:
                        stream.close()


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
    assert calls[1][calls[1].index("--message") + 1] == "What is the session status?"
    assert result["process"]["pid"] == h["record"]["pid"]


@pytest.mark.parametrize("reply", [
    "OK", "OK\n.\n.", "ok", "Ok.", "  OK  ", "**OK**", "`OK`", "\n\nOK\n", "OK!", "OK . .", "\"OK\"",
    # Unicode whitespace and punctuation (fix round 1, minor 6): NBSP, the
    # ideographic space, curly and angle quotes, an ellipsis, a CJK full stop.
    "\u00a0OK\u00a0", "\u3000OK", "\u2018OK\u2019", "\u201cOK\u201d", "\u00abOK\u00bb", "OK\u2026", "OK\u3002",
    "\u201c**OK**\u201d\n.",
])
def test_the_readiness_reply_is_ok_on_its_first_non_empty_line(reply):
    # Qwen3.8-27B answered "OK\n.\n." in the merged-head E2E: the launcher
    # reported NOT READY at random for a brain that had answered.
    assert demo_proof.brain_said_ok(reply) is True


@pytest.mark.parametrize("reply", [
    "", "  \n ", ".", ".\n.", "not ok", "NOT OK", "No.", "OKAY", "I am OK", "OK, but the tools are down",
    "Error: upstream request timed out", "error\nOK", "O.K.",
    "\u2018OK, but the tools are down\u2019", "\u00a0not ok", "\u201cO.K.\u201d", "\u201cOKAY\u201d", "\u00a0\n\u2026",
    "\u201cnot ok\u201d\nOK",
])
def test_anything_but_ok_on_the_first_line_is_not_ready(reply):
    assert demo_proof.brain_said_ok(reply) is False


def test_a_brain_that_answers_ok_with_trailing_dots_passes_the_proof(host_boundary):
    h = host_boundary
    h["reply"] = "OK\n.\n."
    result = demo_proof.run_proof(h["repo"], h["state"], "mujoco", True)
    assert result["verified"] is True


@pytest.mark.parametrize("reply", ["not ok", "", "Error: upstream request timed out"])
def test_a_brain_that_does_not_answer_ok_fails_the_proof_before_any_motion(host_boundary, reply):
    h = host_boundary
    h["reply"] = reply
    with pytest.raises(demo_proof.ProofError, match="did not answer OK"):
        demo_proof.run_proof(h["repo"], h["state"], "mujoco", True)
    assert len(h["calls"]) == 1  # the readiness turn only
    assert json.loads((h["state"] / "proof.json").read_text())["verified"] is False


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
    assert not any(c[c.index("--message") + 1].startswith("Please put the") for c in h["calls"])
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


@pytest.mark.parametrize("prose", [
    "Yes, I am ready.",
    "<tool_call><function=cascade__world_state></function></tool_call>",
])
def test_cosmos_native_probe_rejects_prose_including_xml(monkeypatch, prose):
    check = getattr(demo_proof, "probe_native_tools", None)
    assert callable(check), "OpenClaw bypasses Cosmos3EdgeClient; probe native tool_calls"
    def answer(url, payload=None, timeout=5):
        if url.endswith("/models"):
            return {"data": [{"id": "cosmos3-edge"}]}
        assert {tool["function"]["name"] for tool in payload["tools"]} == {
            "cascade__world_state", "cascade__camera_snapshot",
        }
        return {"choices": [{"message": {"content": prose}}]}
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
                "name": "cascade__world_state", "arguments": '{}',
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


def test_public_generic_proof_uses_natural_robot_requests(host_boundary):
    h = host_boundary
    demo_proof.run_proof(h["repo"], h["state"], "mujoco", True)
    messages = [cmd[cmd.index("--message") + 1] for cmd in h["calls"]]
    assert messages[1:] == [
        "What is the session status?",
        "Please put the red cube in the drop zone.",
        "Let's start over, then show the session status.",
    ]


# ── Nemotron 3.5 Bolt, the default brain ─────────────────────────────────────
# OpenClaw's agent prompt tells the model "Nothing to say: entire reply exactly
# NO_REPLY" and then drops that token. Bolt read "Reply with exactly OK and
# nothing else" as nothing to say (trial proxy seq 5): an empty payload, NOT
# READY. Its readiness question states a reply is required; the judge
# (brain_said_ok) is the same for every brain, and Qwen keeps main's wording.

MAIN_READINESS = "Reply with exactly OK and nothing else. Do not call tools."


def test_readiness_question_is_mains_for_every_brain_but_bolt():
    for model in ("openai/test", "cascade/qwen3.8-27b", "cascade/cosmos3-edge", "nemotron-3.5-bolt", ""):
        assert demo_proof.readiness_prompt(model) == MAIN_READINESS


def test_bolts_readiness_question_says_a_silent_reply_fails():
    prompt = demo_proof.readiness_prompt("cascade/nemotron-3.5-bolt")
    assert prompt != MAIN_READINESS and prompt.endswith(MAIN_READINESS)
    assert "NO_REPLY" in prompt and "visible reply is required" in prompt


@pytest.mark.parametrize("reply", ["NO_REPLY", "", "  "])
def test_a_silent_reply_is_never_ready(reply):
    assert demo_proof.brain_said_ok(reply) is False


def test_the_proof_asks_bolt_its_own_readiness_question(host_boundary):
    h = host_boundary
    h["model"] = "cascade/nemotron-3.5-bolt"
    demo_proof.run_proof(h["repo"], h["state"], "mujoco", False)
    first = h["calls"][0]
    assert first[first.index("--model") + 1] == "cascade/nemotron-3.5-bolt"
    assert first[first.index("--message") + 1] == demo_proof.READINESS_PROMPTS["nemotron-3.5-bolt"]


def test_the_proof_asks_every_other_brain_mains_question(host_boundary):
    h = host_boundary
    demo_proof.run_proof(h["repo"], h["state"], "mujoco", False)
    first = h["calls"][0]
    assert first[first.index("--message") + 1] == MAIN_READINESS


def test_bolt_answering_no_reply_is_still_not_ready(host_boundary):
    h = host_boundary
    h["model"], h["reply"] = "cascade/nemotron-3.5-bolt", "NO_REPLY"
    with pytest.raises(demo_proof.ProofError, match="did not answer OK"):
        demo_proof.run_proof(h["repo"], h["state"], "mujoco", False)


# The robot proof's world and reset turns must execute world_state. Asked "What
# is the session status?", Bolt answered from the tool's description without
# calling it (launch beside Isaac Newton, 2026-10-18): NOT READY. Its questions
# name the tool; the judge (the executed tools) is the same for every brain.

BOLT_WORLD = "What is the session status? Read it with the world_state tool."
BOLT_RESET = "Let's start over, then show the session status with the world_state tool."


def test_the_robot_proof_asks_bolt_questions_that_name_world_state(host_boundary):
    h = host_boundary
    h["model"] = "cascade/nemotron-3.5-bolt"
    result = demo_proof.run_proof(h["repo"], h["state"], "mujoco", True)
    assert result["verified"] is True
    messages = [cmd[cmd.index("--message") + 1] for cmd in h["calls"]]
    assert messages == [demo_proof.READINESS_PROMPTS["nemotron-3.5-bolt"], BOLT_WORLD,
                        "Please put the red cube in the drop zone.", BOLT_RESET]


def test_every_other_brain_keeps_mains_robot_proof_questions():
    for model in ("openai/test", "cascade/qwen3.8-27b", "cascade/cosmos3-edge", "nemotron-3.5-bolt", ""):
        assert demo_proof.proof_prompt(model, demo_proof.WORLD_PROMPT) == "What is the session status?"
        assert (demo_proof.proof_prompt(model, demo_proof.RESET_PROMPT)
                == "Let's start over, then show the session status.")


def test_bolt_skipping_world_state_is_still_not_ready(host_boundary, monkeypatch):
    import subprocess

    h = host_boundary
    h["model"] = "cascade/nemotron-3.5-bolt"
    boundary = demo_proof.subprocess.run

    def answers_without_the_tool(cmd, **kwargs):
        result = boundary(cmd, **kwargs)
        if cmd[0] == "openclaw" and "--message" in cmd and cmd[cmd.index("--message") + 1] == BOLT_WORLD:
            data = json.loads(result.stdout)
            del data["result"]["meta"]["toolSummary"]  # a text reply, as Bolt gave
            return subprocess.CompletedProcess(cmd, 0, json.dumps(data), "")
        return result

    monkeypatch.setattr(demo_proof.subprocess, "run", answers_without_the_tool)
    with pytest.raises(demo_proof.ProofError, match="no valid toolSummary"):
        demo_proof.run_proof(h["repo"], h["state"], "mujoco", True)


def test_auto_brain_resolves_bolt_on_the_local_brain_endpoint(model_http_boundary, monkeypatch):
    h = model_http_boundary
    h["model"] = "nemotron-3.5-bolt"
    monkeypatch.setenv("CASCADE_COSMOS_BASE_URL", "http://127.0.0.1:0/v1")
    monkeypatch.setenv("CASCADE_QWEN_BASE_URL", h["base_url"])
    monkeypatch.delenv("CASCADE_NEMOTRON_BOLT_BASE_URL", raising=False)
    monkeypatch.delenv("CASCADE_INSTALL_PROFILE", raising=False)
    info = demo_proof.check_brain("auto")
    assert info == {"base_url": h["base_url"], "model": "nemotron-3.5-bolt", "brain": "nemotron-bolt",
                    "context_window": 65536}


def test_auto_brain_with_qwen_served_resolves_qwen_exactly_as_before(model_http_boundary, monkeypatch):
    h = model_http_boundary
    h["model"] = "qwen3.8-27b"
    monkeypatch.setenv("CASCADE_COSMOS_BASE_URL", "http://127.0.0.1:0/v1")
    monkeypatch.setenv("CASCADE_QWEN_BASE_URL", h["base_url"])
    monkeypatch.delenv("CASCADE_NEMOTRON_BOLT_BASE_URL", raising=False)
    monkeypatch.delenv("CASCADE_INSTALL_PROFILE", raising=False)
    assert demo_proof.check_brain("auto") == {"base_url": h["base_url"], "model": "qwen3.8-27b", "brain": "qwen",
                                              "context_window": 65536}


def test_an_explicit_bolt_brain_refuses_another_served_model(model_http_boundary, monkeypatch):
    h = model_http_boundary
    h["model"] = "qwen3.8-27b"
    monkeypatch.setenv("CASCADE_NEMOTRON_BOLT_BASE_URL", h["base_url"])
    with pytest.raises(demo_proof.ModelNotServed, match="nemotron-3.5-bolt not served"):
        demo_proof.check_brain("nemotron-bolt")


def test_the_bolt_endpoint_is_its_own_variable_first(model_http_boundary, monkeypatch):
    h = model_http_boundary
    h["model"] = "nemotron-3.5-bolt"
    monkeypatch.setenv("CASCADE_QWEN_BASE_URL", "http://127.0.0.1:0/v1")
    monkeypatch.setenv("CASCADE_NEMOTRON_BOLT_BASE_URL", h["base_url"])
    assert demo_proof.check_brain("nemotron-bolt")["base_url"] == h["base_url"]


def test_the_brain_check_reads_its_bearer_from_a_key_file(monkeypatch, tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "nemotron-3.5-bolt"}]}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        key = tmp_path / "brain.key"
        key.write_text("bolt-key-7c1\n")
        monkeypatch.delenv("CASCADE_MODEL_API_KEY", raising=False)
        monkeypatch.setenv("CASCADE_MODEL_API_KEY_FILE", str(key))
        monkeypatch.setenv("CASCADE_NEMOTRON_BOLT_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
        demo_proof.check_brain("nemotron-bolt")
        assert seen == ["Bearer bolt-key-7c1"]
        monkeypatch.setenv("CASCADE_MODEL_API_KEY", "explicit-key")
        demo_proof.check_brain("nemotron-bolt")
        assert seen[-1] == "Bearer explicit-key"
        monkeypatch.delenv("CASCADE_MODEL_API_KEY")
        monkeypatch.setenv("CASCADE_MODEL_API_KEY_FILE", str(tmp_path / "missing.key"))
        with pytest.raises(demo_proof.ProofError, match="cannot read CASCADE_MODEL_API_KEY_FILE") as raised:
            demo_proof.check_brain("nemotron-bolt")
        assert "missing.key" not in str(raised.value)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_check_brain_cli_accepts_nemotron_bolt(model_http_boundary, monkeypatch, capsys):
    h = model_http_boundary
    h["model"] = "nemotron-3.5-bolt"
    monkeypatch.setenv("CASCADE_NEMOTRON_BOLT_BASE_URL", h["base_url"])
    monkeypatch.delenv("CASCADE_INSTALL_PROFILE", raising=False)
    monkeypatch.setattr(sys, "argv", ["demo_proof.py", "--check-brain", "nemotron-bolt"])
    assert demo_proof.main() in (0, None)
    assert json.loads(capsys.readouterr().out)["model"] == "nemotron-3.5-bolt"


# ── the OpenShell route: OpenClaw lives in the sandbox, the world is long-lived ──

def test_openclaw_commands_can_run_inside_the_sandbox(monkeypatch):
    """On Linux the agent runs in an OpenShell sandbox, so the launcher's
    proof drives the sandbox's own OpenClaw (`openshell_rig.py exec --`);
    a host profile name means nothing there."""
    prefix = [sys.executable, "scripts/openshell_rig.py", "--instance", "sim", "exec", "--", "openclaw"]
    monkeypatch.setenv("CASCADE_OPENCLAW_COMMAND", json.dumps(prefix))
    monkeypatch.setenv("CASCADE_OPENCLAW_PROFILE", "cascade-demo")
    assert demo_proof.oc_command("models", "status", "--json") == [*prefix, "models", "status", "--json"]
    # the default instance is pinned with an empty name, which must survive
    default = [sys.executable, "scripts/openshell_rig.py", "--instance", "", "exec", "--", "openclaw"]
    monkeypatch.setenv("CASCADE_OPENCLAW_COMMAND", json.dumps(default))
    assert demo_proof.oc_command("health") == [*default, "health"]


@pytest.mark.parametrize("bad", ["openclaw", "[]", '["openclaw", 3]', "{}", '[""]'])
def test_a_malformed_openclaw_command_is_refused(monkeypatch, bad):
    monkeypatch.setenv("CASCADE_OPENCLAW_COMMAND", bad)
    with pytest.raises(demo_proof.ProofError, match="CASCADE_OPENCLAW_COMMAND"):
        demo_proof.oc_command("health")


def test_a_persistent_world_is_the_one_server_this_launch_started(host_boundary):
    """The OpenShell route runs ONE long-lived robot server (an HTTP MCP unit
    started before the proof), not a fresh stdio child per session. The
    proof binds to that server: exactly one live MCP of this launch owner."""
    h = host_boundary
    h["binding"] = "persistent"
    record = h["spawn"]()
    result = demo_proof.run_proof(h["repo"], h["state"], "mujoco", True, persistent_mcp=True)
    assert result["verified"] is True
    assert result["process"]["pid"] == record["pid"]
    assert result["trace"] == str(h["trace"])


@pytest.mark.parametrize("binding", ["none", "two", "foreign", "dead"])
def test_a_persistent_world_must_be_exactly_this_launchs_live_server(host_boundary, binding):
    h = host_boundary
    h["binding"] = "persistent"
    if binding == "two":
        h["spawn"](), h["spawn"]()
    elif binding == "foreign":
        from cascade.apps import process_owner as owners
        h["spawn"](owners.load_owner(h["state"].parent / "foreign", h["repo"], "foreign", create=True))
    elif binding == "dead":
        h["spawn"](dead=True)
    with pytest.raises(demo_proof.ProofError, match="live MCP"):
        demo_proof.run_proof(h["repo"], h["state"], "mujoco", True, persistent_mcp=True)
    assert not any(c[c.index("--message") + 1].startswith("Please put the") for c in h["calls"])


def test_without_persistence_an_older_server_still_cannot_be_the_world(host_boundary):
    """The stdio route keeps its rule: the world must be NEW for this proof."""
    h = host_boundary
    h["binding"] = "persistent"
    h["spawn"]()
    with pytest.raises(demo_proof.ProofError, match="NEW"):
        demo_proof.run_proof(h["repo"], h["state"], "mujoco", True)


def test_the_cli_selects_the_persistent_world(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(demo_proof, "run_proof", lambda *a, **kw: seen.update(kw) or {"verified": True})
    monkeypatch.setattr(sys, "argv", ["demo_proof.py", "--run", "--state-dir", str(tmp_path), "--persistent-mcp"])
    assert demo_proof.main() == 0
    assert seen["persistent_mcp"] is True


def test_native_probe_requests_diagnostic_evidence_without_forcing_tool_choice(model_http_boundary):
    boundary = model_http_boundary
    result = demo_proof.probe_native_tools(boundary["base_url"], "cosmos3-edge")
    assert result["native_tools"] is True
    request = boundary["requests"][-1]
    assert request["tool_choice"] == "auto"
    assert len(request["tools"]) == 2
    prompt = request["messages"][-1]["content"].lower()
    assert prompt == "what objects does the tracked world model currently contain?"
    assert all(tool["function"]["name"] not in prompt for tool in request["tools"])
    assert boundary["requests"][0] == "/v1/models"
    assert len(boundary["requests"]) == 2  # GET model + generation; no tool execution.
