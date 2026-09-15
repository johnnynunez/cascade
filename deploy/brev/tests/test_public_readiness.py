"""A registered tunnel is ready only after its authenticated visitor responds."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import threading
import time

import pytest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_readiness", HERE / "public.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
ORIGIN = "https://visitor.example"
AUTH = {"username": "visitor", "password": "test-password"}


def tunnel(url=ORIGIN, address=installer.VISITOR_ORIGIN):
    return {"proto": "https", "public_url": url, "config": {"addr": address}}


def test_unrelated_tunnel_is_not_returned_as_the_visitor():
    unrelated = tunnel("https://unrelated.example", "http://127.0.0.1:8092")
    assert installer.visitor_url([unrelated]) is None
    assert installer.visitor_url([unrelated, tunnel()]) == ORIGIN
    with pytest.raises(ValueError, match="More than one"):
        installer.visitor_url([tunnel(), tunnel("https://another.example")])


@pytest.mark.parametrize("url", ["http://visitor.example", "https://user:password@visitor.example",
                                  ORIGIN + "/openclaw", ORIGIN + "?token=fixture"])
def test_tunnel_must_return_a_plain_https_origin(url):
    with pytest.raises(ValueError, match="invalid HTTPS origin"):
        installer.visitor_url([tunnel(url)])


@pytest.fixture
def public_responses(monkeypatch):
    cameras = [{"name": name, "online": True, "frame_id": 42}
               for name in ("kitchen", "worktop", "side")]
    responses = {
        ("/", False): (401, b"Authentication required"),
        ("/", True): (200, (HERE / "visitor.html").read_bytes()),
        ("/staff/", False): (401, b"Authentication required"),
        ("/staff/", True): (200, (HERE / "staff.html").read_bytes()),
        ("/staff/style.css", True): (200, (HERE / "staff.css").read_bytes()),
        ("/api/status", True): (200, json.dumps({"cameras": cameras}).encode()),
        ("/snapshot/worktop.jpg", True): (200, b"\xff\xd8fixture\xff\xd9"),
        **{(path, True): (404, b"Not found") for path in
           ("/openclaw/", "/api/openclaw-bootstrap", "/__openclaw/", "/api/control")},
    }
    seen = []

    def get(url, authorization, deadline):
        key = (url.removeprefix(ORIGIN), bool(authorization))
        seen.append(key)
        assert deadline > time.monotonic()
        return responses[key]

    monkeypatch.setattr(installer, "public_get", get)
    return responses, seen


def test_readiness_requires_page_cameras_authentication_and_admin_exclusion(public_responses):
    responses, seen = public_responses
    installer.verify_visitor(ORIGIN, AUTH)
    assert set(seen) == set(responses)


@pytest.mark.parametrize("path,authenticated,response,message", [
    ("/", False, (200, b"Public page"), "require authentication"),
    ("/", True, (200, b"Different application"), "installed visitor page"),
    ("/staff/", False, (200, b"Public staff page"), "require authentication"),
    ("/staff/", True, (200, b"Attendee page"), "installed staff page"),
    ("/staff/style.css", True, (404, b"Missing stylesheet"), "installed staff page"),
    ("/api/status", True, (200, b'{"cameras":[]}'), "three public cameras"),
    ("/snapshot/worktop.jpg", True, (200, b"\xff\xd8truncated"), "complete JPEG"),
    ("/openclaw/", True, (200, b"Admin"), "administrative routes"),
])
def test_registered_tunnel_cannot_hide_a_failed_visitor_check(
        public_responses, path, authenticated, response, message):
    responses, _ = public_responses
    responses[(path, authenticated)] = response
    with pytest.raises(RuntimeError, match=message):
        installer.verify_visitor(ORIGIN, AUTH)


def test_verification_never_follows_a_redirect_with_visitor_credentials():
    requests = []

    class Redirect(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests.append(self.path)
            self.send_response(302)
            self.send_header("Location", "/credential-recipient")
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _ = installer.public_get(f"http://127.0.0.1:{server.server_port}/",
                                         "Basic fixture", time.monotonic() + 3)
        assert status == 302
        assert requests == ["/"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_verification_deadline_expires_before_another_request():
    with pytest.raises(TimeoutError, match="verification timed out"):
        installer.public_get(ORIGIN, "Basic fixture", time.monotonic() - 1)
