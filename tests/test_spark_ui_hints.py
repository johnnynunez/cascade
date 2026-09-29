"""Exercise the real HTTP handler on ephemeral ports; never bind the demo ports."""
import base64
from contextlib import contextmanager
import http.client
import threading

import pytest

from test_spark_visitor_chat import chat as _chat, visitor

chat = _chat
PATHS = ("/guide", "/guide/", "/openclaw", "/openclaw/")


@contextmanager
def serving(chat, *, port, authorization=""):
    server = visitor.VisitorServer(("127.0.0.1", 0), visitor.VisitorHandler)
    actual_port = server.server_port
    # Logical deployment port for routing; the socket remains ephemeral.
    server.server_port = port
    server.chat = chat
    server.authorization = authorization
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    worker.start()
    try:
        yield actual_port
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def get(port, path, **headers):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def assert_private(headers, body):
    response = repr(headers).encode() + body
    assert b"private-token-fixture" not in response
    assert b"#token=" not in response
    assert b"18790" not in response


def test_local_guide_redirects_and_openclaw_explains_command_without_credentials(chat):
    with serving(chat, port=8092) as port:
        for path in PATHS:
            for query in ("", "?connect=1&token=private-token-fixture&next=https://example.invalid"):
                code, headers, body = get(port, path + query, Host="127.0.0.1:8092")
                assert_private(headers, body)
                if path.startswith("/guide"):
                    assert code == 302 and headers["Location"] == "/" and not body
                else:
                    assert code == 200 and "Location" not in headers
                    assert b"./run.sh dashboard" in body and b"--no-open" in body
                    assert headers["Content-Type"] == "text/html; charset=utf-8"
        for path in ("/", "/api/chat", "/visitor.js"):
            code, headers, body = get(port, path, Host="127.0.0.1:8092")
            assert code == 200
            assert_private(headers, body)
        assert get(port, "/guide", Host="attacker.invalid:8092")[0] == 403


@pytest.mark.parametrize("host", ["127.0.0.1:8093", "visitor.example.ngrok.app", "127.0.0.1:8092"])
@pytest.mark.parametrize("authenticated", [False, True])
def test_public_visitor_and_ngrok_hosts_never_serve_operator_hints_or_token(chat, host, authenticated):
    authorization = "Basic " + base64.b64encode(b"visitor:fixture-password").decode()
    with serving(chat, port=8093, authorization=authorization) as port:
        request_headers = {"Host": host, "X-Forwarded-Host": "127.0.0.1:8092", "X-Forwarded-Port": "8092"}
        if authenticated:
            request_headers["Authorization"] = authorization
        for path in PATHS + ("/api/openclaw-bootstrap", "/openclaw/openclaw.json", "/config"):
            code, headers, body = get(port, path + "?token=private-token-fixture", **request_headers)
            assert code == (404 if authenticated else 401)
            assert "Location" not in headers and b"./run.sh dashboard" not in body
            assert_private(headers, body)
        for path in ("/", "/api/chat", "/visitor.js"):
            code, headers, body = get(port, path, **request_headers)
            assert code == (200 if authenticated else 401)
            assert_private(headers, body)


@pytest.mark.parametrize("port,with_chat,authorization", [(8093, True, ""), (8092, False, ""), (8092, True, "Basic fixture")])
def test_local_hints_require_the_local_unauthenticated_chat_surface(chat, port, with_chat, authorization):
    with serving(chat if with_chat else None, port=port, authorization=authorization) as actual:
        for path in PATHS:
            assert get(actual, path, Host=f"127.0.0.1:{port}", Authorization=authorization)[0] == 404
