"""scripts/brain_v1_live_check.js: the one-minute live check of the brain's OpenShell /v1-only profile.

It runs inside the sandbox (`openshell_rig.py exec -- node -e "$(cat scripts/brain_v1_live_check.js)" <served id>`);
here it runs on the host's node against a fake brain on a private loopback port that plays the OpenShell proxy (or a
broken one). #405 review, Minor 3: the old check used fetch(), which normalizes `/v1/../slots` and `/v1/%2e%2e/slots`
to `/slots` before sending, so those rows proved nothing; the fake records the raw request target to pin that the dot
paths really leave the client as written. Nothing here reaches a live port.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from owned_server import held_dead_port

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "brain_v1_live_check.js"
KEY = "openshell:resolve:env:CASCADE_BRAIN_API_KEY-test-7f3"
MODEL = "nemotron-3.5-bolt"
#: The rows in order: (method, raw request target). The dot rows must leave the client exactly like this.
ROWS = [("GET", "/v1/models"), ("POST", "/v1/chat/completions"), ("POST", "/v1/chat/completions"),
        ("GET", "/slots"), ("GET", "/props"), ("GET", "/metrics"), ("POST", "/tokenize"),
        ("GET", "/v1/../slots"), ("GET", "/v1/%2e%2e/slots")]
DENIED = json.dumps({"status": 403, "error": "policy_denied", "policy": "cascade-brain", "rule": "GET /slots"})


class FakeBrain:
    """A loopback HTTP server that answers like the OpenShell proxy in front of the brain (`mode="proxy"`) or like
    a broken route, and records every request's method, raw target, headers and body."""

    def __init__(self, mode: str = "proxy"):
        self.mode, self.requests = mode, []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def answer(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                fake.requests.append({"method": self.command, "path": self.path,
                                      "headers": {k.lower(): v for k, v in self.headers.items()},
                                      "body": json.loads(body) if body else None})
                status, ctype, text = fake.reply(self.command, self.path, json.loads(body) if body else {})
                data = text.encode()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = answer

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def reply(self, method: str, path: str, body: dict) -> tuple[int, str, str]:
        v1 = path.startswith("/v1/") and ".." not in path and "%2e" not in path.lower()
        if v1 and self.mode == "unswapped":
            return 401, "application/json", '{"error":{"message":"Invalid API Key","code":401}}'
        if v1 and self.mode == "v1_denied":   # a profile that blocks the agent's own API too
            return 403, "application/json", DENIED
        if v1 and path == "/v1/models":
            return 200, "application/json", json.dumps({"data": [{"id": MODEL}]})
        if v1 and path == "/v1/chat/completions":
            if body.get("stream") and self.mode != "stream_as_json":
                return 200, "text/event-stream", 'data: {"choices":[{"delta":{"content":"OK"}}]}\n\ndata: [DONE]\n\n'
            return 200, "application/json", json.dumps({"choices": [{"message": {"content": "OK"}}]})
        if self.mode == "open":   # the profile is not in force: the brain itself answers
            return 200, "application/json", '{"slots":[{"prompt":"secret prompt"}]}'
        if self.mode == "dots_400" and ("/../" in path or "%2e%2e" in path.lower()):
            return 400, "text/plain", "Bad Request"
        if self.mode == "tokenize_400" and path == "/tokenize":   # vLLM's own validation error: it got there
            return 400, "application/json", '{"error":{"message":"Field required: prompt","code":400}}'
        if self.mode == "server_403":
            return 403, "application/json", '{"error":"forbidden"}'
        return 403, "application/json", DENIED

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def node() -> str:
    path = shutil.which("node")
    if not path:
        pytest.skip("node not on PATH")
    return path


def check(port: int | None, *args: str, form: str = "file", **env) -> subprocess.CompletedProcess:
    e = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""), "CASCADE_BRAIN_API_KEY": KEY,
         "CASCADE_BRAIN_HOST": "127.0.0.1", **({} if port is None else {"CASCADE_BRAIN_PORT": str(port)}), **env}
    e = {k: v for k, v in e.items() if v is not None}
    argv = [node(), str(SCRIPT), *args] if form == "file" else [node(), "-e", SCRIPT.read_text(), *args]
    return subprocess.run(argv, env=e, capture_output=True, text=True, timeout=60)


def verdicts(result) -> list[str]:
    """PASS / FAIL per row, in row order."""
    return [line.split()[0] for line in result.stdout.splitlines() if line.split()[:1] in (["PASS"], ["FAIL"])]


@pytest.mark.parametrize("form", ["file", "node -e"])
def test_a_v1_only_proxy_passes_every_row_and_the_dot_paths_leave_the_client_raw(form):
    with FakeBrain() as brain:
        r = check(brain.port, MODEL, form=form)
    assert r.returncode == 0, r.stdout + r.stderr
    assert verdicts(r) == ["PASS"] * 9 and "9/9 rows pass" in r.stdout
    assert [(q["method"], q["path"]) for q in brain.requests] == ROWS
    assert {q["headers"]["authorization"] for q in brain.requests} == {f"Bearer {KEY}"}
    chats = [q["body"] for q in brain.requests if q["path"] == "/v1/chat/completions"]
    assert [(c["model"], c["stream"]) for c in chats] == [(MODEL, False), (MODEL, True)]
    assert all(c["max_tokens"] <= 16 for c in chats), "a live check must not keep the brain busy"


def test_a_brain_answering_outside_v1_fails_those_rows():
    """The sandbox reached /slots, /props, /metrics, /tokenize: the profile is not in force."""
    with FakeBrain("open") as brain:
        r = check(brain.port, MODEL)
    assert r.returncode == 1
    assert verdicts(r) == ["PASS"] * 3 + ["FAIL"] * 6
    assert "6 of 9 rows FAIL" in r.stdout and "not in force" in r.stdout


def test_the_dot_rows_also_pass_on_400():
    with FakeBrain("dots_400") as brain:
        r = check(brain.port, MODEL)
    assert r.returncode == 0 and verdicts(r) == ["PASS"] * 9, r.stdout


def test_a_400_passes_only_on_the_dot_rows():
    """vLLM answers POST /tokenize {"content": ...} with its own 400 (it wants "prompt"): that request reached it."""
    with FakeBrain("tokenize_400") as brain:
        r = check(brain.port, MODEL)
    assert r.returncode == 1 and verdicts(r) == ["PASS"] * 6 + ["FAIL", "PASS", "PASS"], r.stdout


def test_a_403_that_is_not_the_proxys_policy_denied_fails():
    """Only OpenShell's denial counts: a 403 from anything else means the request got past the proxy."""
    with FakeBrain("server_403") as brain:
        r = check(brain.port, MODEL)
    assert r.returncode == 1 and verdicts(r) == ["PASS"] * 3 + ["FAIL"] * 6
    assert "policy_denied" in r.stdout


def test_an_unswapped_key_fails_the_v1_rows_and_says_why():
    with FakeBrain("unswapped") as brain:
        r = check(brain.port, MODEL)
    assert r.returncode == 1 and verdicts(r) == ["FAIL"] * 3 + ["PASS"] * 6
    assert "placeholder" in r.stdout


def test_a_profile_that_denies_the_agents_own_api_fails_the_v1_rows():
    with FakeBrain("v1_denied") as brain:
        r = check(brain.port, MODEL)
    assert r.returncode == 1 and verdicts(r) == ["FAIL"] * 3 + ["PASS"] * 6, r.stdout


def test_a_stream_that_is_not_server_sent_events_fails():
    with FakeBrain("stream_as_json") as brain:
        r = check(brain.port, MODEL)
    assert r.returncode == 1 and verdicts(r) == ["PASS", "PASS", "FAIL"] + ["PASS"] * 6
    assert "text/event-stream" in r.stdout


def test_a_brain_that_does_not_answer_fails_every_row():
    with held_dead_port() as port:
        r = check(port, MODEL)
    assert r.returncode == 1 and verdicts(r) == ["FAIL"] * 9


@pytest.mark.parametrize("missing", ["served id", "key"])
def test_a_missing_served_id_or_key_is_a_usage_error_before_any_request(missing):
    with FakeBrain() as brain:
        r = check(brain.port, *(() if missing == "served id" else (MODEL,)),
                  **({"CASCADE_BRAIN_API_KEY": None} if missing == "key" else {}))
    assert r.returncode == 2 and brain.requests == []
    assert ("CASCADE_BRAIN_API_KEY" if missing == "key" else "served id") in r.stderr


def test_the_target_defaults_to_the_sandboxs_brain_alias_and_port():
    """Inside the sandbox the brain is host.openshell.internal on $CASCADE_BRAIN_PORT (set by `up`), else 8081.
    Read from the usage error, so no request leaves this host."""
    r = check(None, CASCADE_BRAIN_HOST=None)
    assert r.returncode == 2 and "http://host.openshell.internal:8081" in r.stderr
    r = check(18123, CASCADE_BRAIN_HOST=None)
    assert "http://host.openshell.internal:18123" in r.stderr
