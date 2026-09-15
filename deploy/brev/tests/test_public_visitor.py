"""The public HTTP surface cannot forward administrative routes or credentials."""
import base64
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import subprocess
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

HERE = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location("public_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


visitor = load("visitor")
installer = load("public")


@contextmanager
def serving(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def fetch(url, *, headers=None, method="GET"):
    try:
        response = urlopen(Request(url, headers=headers or {}, method=method), timeout=3)
    except HTTPError as error:
        response = error
    with response:
        return response.status, response.read()


@pytest.fixture
def camera():
    requests = []

    class Camera(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests.append(self.path)
            state = {"private_token": "never-public", "cameras": {
                "worktop": {"online": True, "frame_id": 42, "fps": 3, "private_token": "never-public"}}}
            body = json.dumps(state).encode() if self.path == "/state" else b"\xff\xd8fixture\xff\xd9"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with serving(Camera) as (_, origin):
        yield origin, requests


def test_administrative_routes_and_writes_never_reach_the_camera_backend(camera):
    with serving(visitor.VisitorHandler) as (server, origin):
        server.authorization = ""
        server.camera_origin = camera[0]
        for path in ("/openclaw/", "/api/openclaw-bootstrap", "/__openclaw/", "/api/control",
                     "/snapshot/../private", "/%2e%2e/openclaw/", "/staff/../openclaw/",
                     "/staff/%2e%2e/openclaw/", "/staff/auth.json", "/staff/index.html"):
            assert fetch(origin + path)[0] == 404
        assert fetch(origin + "/api/status", method="POST")[0] == 405
    assert camera[1] == []


def test_public_status_is_a_camera_allowlist_and_query_cannot_change_upstream(camera):
    with serving(visitor.VisitorHandler) as (server, origin):
        server.authorization = ""
        server.camera_origin = camera[0]
        status, body = fetch(origin + "/api/status")
        assert status == 200 and b"never-public" not in body
        assert json.loads(body)["cameras"][1]["frame_id"] == 42
        status, body = fetch(origin + "/snapshot/worktop.jpg?url=http://private.invalid/openclaw/")
        assert status == 200 and body.startswith(b"\xff\xd8")
    assert camera[1] == ["/state", "/snapshot/worktop.jpg"]


def test_local_basic_auth_protects_the_entry_page(camera):
    with serving(visitor.VisitorHandler) as (server, origin):
        server.camera_origin = camera[0]
        server.authorization = "Basic " + base64.b64encode(b"visitor:fixture-password").decode()
        assert fetch(origin + "/")[0] == 401
        assert fetch(origin + "/", headers={"Authorization": "Basic wrong"})[0] == 401
        status, body = fetch(origin + "/", headers={"Authorization": server.authorization})
        assert status == 200 and b"Live kitchen" in body


def test_configuration_timeout_does_not_disclose_the_token(monkeypatch):
    token = "private-token-fixture"
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 1, output=token)
    monkeypatch.setattr(installer.subprocess, "run", timeout)
    with pytest.raises(RuntimeError) as raised:
        installer.run(["ngrok", "config", "add-authtoken", token])
    assert token not in str(raised.value)


def test_staff_page_is_separate_authenticated_and_cannot_proxy_requests(camera):
    with serving(visitor.VisitorHandler) as (server, origin):
        server.camera_origin = camera[0]
        server.authorization = "Basic " + base64.b64encode(b"staff:fixture-password").decode()
        headers = {"Authorization": server.authorization}
        for path, name in (("/staff/", "staff.html"), ("/staff/style.css", "staff.css")):
            assert fetch(origin + path)[0] == 401
            assert fetch(origin + path, headers={"Authorization": "Basic wrong"})[0] == 401
            status, body = fetch(origin + path + "?url=http://private.invalid/openclaw/", headers=headers)
            assert status == 200 and body == (HERE / name).read_bytes()
        assert fetch(origin + "/staff/", headers=headers)[1] != fetch(origin + "/", headers=headers)[1]
        assert fetch(origin + "/staff/", headers=headers, method="POST")[0] == 405
    assert camera[1] == []


def test_video_auth_and_camera_allowlist_precede_encoder_access(camera):
    class Streams:
        def get(self, name):
            pytest.fail("Unauthorized or invalid video request started an encoder")
    with serving(visitor.VisitorHandler) as (server, origin):
        server.camera_origin = camera[0]
        server.authorization = "Basic fixture"
        server.video = Streams()
        assert fetch(origin + "/video/worktop.mp4")[0] == 401
        assert fetch(origin + "/video/../private.mp4", headers={"Authorization": server.authorization})[0] == 404
        assert fetch(origin + "/video/unknown.mp4", headers={"Authorization": server.authorization})[0] == 404
        server.video = None
        assert fetch(origin + "/video/worktop.mp4", headers={"Authorization": server.authorization})[0] == 404


def test_incomplete_camera_jpeg_is_not_presented(camera, monkeypatch):
    class Response(BytesIO):
        pass
    monkeypatch.setattr(visitor, "urlopen", lambda *args, **kwargs: Response(b"\xff\xd8partial"))
    with serving(visitor.VisitorHandler) as (server, origin):
        server.authorization = ""
        server.camera_origin = camera[0]
        assert fetch(origin + "/snapshot/worktop.jpg")[0] == 503


def test_connection_limit_refuses_excess_clients_and_releases_capacity():
    entered = threading.Event()
    release = threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            if self.path == "/hold":
                entered.set()
                release.wait(timeout=3)
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
    server = visitor.VisitorServer(("127.0.0.1", 0), Handler, max_connections=1)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    held = threading.Thread(target=lambda: fetch(origin + "/hold"))
    held.start()
    try:
        assert entered.wait(timeout=2)
        assert fetch(origin + "/")[0] == 503
        release.set()
        held.join(timeout=2)
        assert not held.is_alive()
        assert fetch(origin + "/")[0] == 200
    finally:
        release.set()
        held.join(timeout=2)
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
