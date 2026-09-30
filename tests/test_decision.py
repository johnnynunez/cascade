"""Protocol tests use an HTTP stub, not Jev/Kev or a model-quality estimate."""

from __future__ import annotations

import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import pytest

from cascade.agent.decision import (
    ChoiceAnswer, ChoiceQuestion, DecisionError, NoulAnswer, NoulQuestion,
    SystemOneClient, validate_response,
)


QUESTIONS = {
    "next": ChoiceQuestion("Which next skill?", {"observe": "Look first", "defer": "Ask for a plan"}),
    "complete": NoulQuestion("Is the goal independently verified?"),
}
RESPONSE = {
    "model": "stub-model-for-tests",
    "answers": {
        "next": {"type": "choice", "choice": "observe",
                 "probabilities": {"observe": 0.8, "defer": 0.2}, "confidence": 0.4},
        "complete": {"type": "noul", "noul": 0.1},
    },
    "usage": {"input_tokens": 31, "output_tokens": 4},
}


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            self.server.received.append({"path": self.path, "body": json.loads(body),
                                         "headers": dict(self.headers)})
            if self.server.delay:
                time.sleep(self.server.delay)
            self.send_response(self.server.reply_status)
            if self.server.redirect:
                self.send_header("Location", self.server.redirect)
            if self.server.reply_chunked:
                self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            try:
                self.wfile.write(self.server.reply)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.received = []
    httpd.reply = json.dumps(RESPONSE).encode()
    httpd.reply_status = 200
    httpd.redirect = None
    httpd.delay = 0
    httpd.reply_chunked = False
    httpd.origin = f"http://127.0.0.1:{httpd.server_port}"
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=2)


def test_native_request_and_typed_response_no_env_key_leak(server, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-secret-never-sent-to-localhost")
    client = SystemOneClient(server.origin, "explicit-alias")
    result = client.evaluate({"request": "Observa la escena", "holding": None}, QUESTIONS)
    assert result.requested_model == "explicit-alias"
    assert result.reported_model == "stub-model-for-tests"
    assert result.answers["next"] == ChoiceAnswer("observe", {"observe": 0.8, "defer": 0.2}, 0.4)
    assert result.answers["complete"] == NoulAnswer(0.1)
    assert result.usage == {"input_tokens": 31, "output_tokens": 4}
    assert result.latency_ms >= 0
    assert len(result.request_sha256) == len(result.response_sha256) == 64
    assert len(server.received) == 1
    sent = server.received[0]
    assert sent["path"] == "/v1/systemone"
    assert sent["body"] == {"model": "explicit-alias", "state": {"request": "Observa la escena", "holding": None},
                             "questions": {k: q.to_wire() for k, q in QUESTIONS.items()}}
    assert "Authorization" not in sent["headers"]


@pytest.mark.parametrize("origin", [
    "http://example.com", "http://api.typesafe.ai", "https://api.typesafe.ai.evil.test",
    "https://user:secret@api.typesafe.ai", "https://api.typesafe.ai:444",
    "https://api.typesafe.ai?key=secret", "https://api.typesafe.ai/v1",
    "http://127.0.0.1:28009",
])
def test_credentials_are_bound_to_official_https_origin(origin):
    with pytest.raises(ValueError):
        SystemOneClient(origin, "jev-latest", api_key="test-only-key")


def test_official_key_header_is_explicit_and_never_in_payload(monkeypatch):
    import io

    class Reply(io.BytesIO):
        status = 200

    class Opener:
        def open(self, request, *, timeout):
            assert request.full_url == "https://api.typesafe.ai/v1/systemone"
            assert request.get_header("Authorization") == "Bearer test-only-key"
            assert "test-only-key" not in request.data.decode()
            assert timeout == 2.0
            return Reply(json.dumps(RESPONSE).encode())

    client = SystemOneClient("https://api.typesafe.ai", "jev-latest", api_key="test-only-key", timeout_s=2)
    monkeypatch.setattr(client, "_opener", Opener())
    assert client.evaluate("state", QUESTIONS).reported_model == "stub-model-for-tests"


@pytest.mark.parametrize("status", [301, 302, 307, 308, 401, 422, 429, 500, 529])
def test_errors_and_redirects_never_retry_or_echo_body(server, status):
    server.reply_status = status
    server.redirect = server.origin + "/redirect-target"
    server.reply = b'{"error": "echoed-secret-or-private-state"}'
    client = SystemOneClient(server.origin, "test")
    with pytest.raises(DecisionError, match=f"HTTP {status}") as caught:
        client.evaluate("state", QUESTIONS)
    assert "echoed-secret" not in str(caught.value)
    assert len(server.received) == 1


def test_timeout_is_an_explicit_error_without_retry(server):
    server.delay = 0.2
    client = SystemOneClient(server.origin, "test", timeout_s=0.02)
    with pytest.raises(DecisionError, match="timed out; no retry"):
        client.evaluate("state", QUESTIONS)
    assert len(server.received) == 1


def test_truncated_chunked_response_is_an_explicit_error_without_retry(server):
    server.reply_chunked = True
    # The second chunk promises five bytes but the connection ends after one.
    server.reply = b"3\r\nabc\r\n5\r\nx"
    with pytest.raises(DecisionError, match="connection failed.*no retry"):
        SystemOneClient(server.origin, "test").evaluate("state", QUESTIONS)
    assert len(server.received) == 1


@pytest.mark.parametrize("value", [pytest.param(10**400, id="huge_positive_integer"),
                                   pytest.param(-(10**400), id="huge_negative_integer")])
def test_large_valid_json_integer_probability_is_a_protocol_error(server, value):
    payload = copy.deepcopy(RESPONSE)
    payload["answers"]["complete"]["noul"] = value
    server.reply = json.dumps(payload).encode()
    with pytest.raises(DecisionError, match="finite and between 0 and 1"):
        SystemOneClient(server.origin, "test").evaluate("state", QUESTIONS)
    assert len(server.received) == 1


@pytest.mark.parametrize("raw", [
    b"not-json", b"null", b"[]", b'\xff',
    b'{"model":"test","model":"other","answers":{}}',
    b'{"model":"test","answers":{},"extra":NaN}',
    b'{"model":"test","answers":{},"extra":Infinity}',
])
def test_malformed_json_is_rejected(server, raw):
    server.reply = raw
    with pytest.raises(DecisionError):
        SystemOneClient(server.origin, "test").evaluate("state", QUESTIONS)


@pytest.mark.parametrize("value", [None, True, "0.8", -0.01, 1.01, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["noul", "confidence", "probability"])
def test_probability_fields_reject_invalid_values(value, field):
    payload = copy.deepcopy(RESPONSE)
    if field == "noul":
        payload["answers"]["complete"]["noul"] = value
    elif field == "confidence":
        payload["answers"]["next"]["confidence"] = value
    else:
        payload["answers"]["next"]["probabilities"]["observe"] = value
    with pytest.raises(DecisionError):
        validate_response(payload, QUESTIONS)


@pytest.mark.parametrize("mutation", [
    lambda p: p.pop("model"),
    lambda p: p.update(model=""),
    lambda p: p.update(model=True),
    lambda p: p["answers"].pop("complete"),
    lambda p: p["answers"].update(extra={"type": "noul", "noul": 0.3}),
    lambda p: p["answers"]["next"].update(type="noul"),
    lambda p: p["answers"]["complete"].update(type="choice"),
    lambda p: p["answers"]["next"].pop("confidence"),
    lambda p: p["answers"]["next"].update(choice="execute_unlisted_action"),
    lambda p: p["answers"]["next"].update(choice="defer"),
    lambda p: p["answers"]["next"].update(probabilities={"observe": 1.0}),
    lambda p: p["answers"]["next"].update(probabilities={"observe": 0.5, "defer": 0.4, "extra": 0.1}),
    lambda p: p["answers"]["next"].update(probabilities={"observe": 0.8, "defer": 0.8}),
    lambda p: p.update(usage={"input_tokens": True}),
    lambda p: p.update(usage={"input_tokens": -1}),
])
def test_response_contract_is_fail_closed(mutation):
    payload = copy.deepcopy(RESPONSE)
    mutation(payload)
    with pytest.raises(DecisionError):
        validate_response(payload, QUESTIONS)


def test_tied_argmax_is_valid_and_probabilities_are_not_renormalized():
    payload = copy.deepcopy(RESPONSE)
    payload["answers"]["next"].update(choice="defer", probabilities={"observe": 0.5, "defer": 0.5})
    _, answers, _ = validate_response(payload, QUESTIONS)
    assert answers["next"].probabilities == {"observe": 0.5, "defer": 0.5}
    assert answers["next"].choice == "defer"


@pytest.mark.parametrize("total_delta", [-0.0005, -0.0001, 0.0001, 0.0005])
def test_four_decimal_server_rounding_is_preserved_for_eleven_options(total_delta):
    criteria = {f"skill_{i}": None for i in range(11)}
    probabilities = {name: 0.0 for name in criteria}
    probabilities.update(skill_0=0.6, skill_1=round(0.4 + total_delta, 4))
    payload = {"model": "rounded-server-stub", "answers": {
        "next": {"type": "choice", "choice": "skill_0",
                 "probabilities": probabilities, "confidence": 0.2},
    }}
    _, answers, _ = validate_response(payload, {"next": ChoiceQuestion("Next?", criteria)})
    assert answers["next"].probabilities == probabilities
    assert sum(answers["next"].probabilities.values()) == pytest.approx(1 + total_delta)


@pytest.mark.parametrize("total_delta", [-0.0006, 0.0006])
def test_eleven_option_sum_beyond_four_decimal_rounding_bound_is_rejected(total_delta):
    criteria = {f"skill_{i}": None for i in range(11)}
    probabilities = {name: 0.0 for name in criteria}
    probabilities.update(skill_0=0.6, skill_1=round(0.4 + total_delta, 4))
    payload = {"model": "rounded-server-stub", "answers": {
        "next": {"type": "choice", "choice": "skill_0",
                 "probabilities": probabilities, "confidence": 0.2},
    }}
    with pytest.raises(DecisionError, match="four-decimal rounding error"):
        validate_response(payload, {"next": ChoiceQuestion("Next?", criteria)})


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf")])
def test_timeout_configuration_is_validated(timeout):
    with pytest.raises(ValueError):
        SystemOneClient("http://127.0.0.1:28009", "test", timeout_s=timeout)


@pytest.mark.parametrize("state,questions", [
    (True, QUESTIONS),
    ({"nonfinite": float("nan")}, QUESTIONS),
    ("state", {}),
    ("state", {"": NoulQuestion("yes?")}),
    ("state", {"q": {"type": "noul"}}),
    ("state", {"q": ChoiceQuestion("", {"a": "a", "b": "b"})}),
    ("state", {"q": ChoiceQuestion("choose", {"a": "a"})}),
    ("state", {"q": ChoiceQuestion("choose", {"a": 1, "b": "b"})}),
    ("state", {"q": NoulQuestion("")}),
])
def test_invalid_requests_never_reach_network(server, state, questions):
    with pytest.raises(ValueError):
        SystemOneClient(server.origin, "test").evaluate(state, questions)
    assert server.received == []
