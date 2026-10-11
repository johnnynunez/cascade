"""The dashboard chat (POST /task) moves the arm; from the LAN it must not.

The live view binds 0.0.0.0 on purpose: the big screen and the staff phone's
STOP button (POST /cancel) reach it from other machines. But the same server
also hands POST /task to the reflex grammar ("home", "wave", ...) with no auth,
so before this anyone on the venue Wi-Fi could drive the real arm. The policy:
stopping is always allowed from anywhere; commanding motion is allowed only
from this machine unless the operator opts in (`stream.lan_commands: true`).
"""
from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request

import pytest

from cascade.apps.stream_server import StreamServer, command_allowed


class _Rig:
    """No cameras: these tests only exercise the POST routes."""

    names = ()


def _lan_ip() -> str:
    """This host's primary non-loopback IPv4 (a UDP connect sends nothing)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))
        ip = s.getsockname()[0]
    except OSError:
        pytest.skip("no non-loopback IPv4 to act as a LAN client")
    finally:
        s.close()
    if ip.startswith("127."):
        pytest.skip("no non-loopback IPv4 to act as a LAN client")
    return ip


def _post(url: str, body: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(body or {}).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.53", "::1", "::ffff:127.0.0.1"])
def test_loopback_clients_may_command(host):
    assert command_allowed(host, lan_commands=False)


@pytest.mark.parametrize("host", ["192.168.1.50", "172.18.0.2", "10.0.0.7", "fe80::1", "", "garbage"])
def test_lan_clients_may_not_command_by_default(host):
    assert not command_allowed(host, lan_commands=False)


def test_operator_can_opt_in_to_lan_commands():
    assert command_allowed("192.168.1.50", lan_commands=True)


@pytest.fixture
def lan_server():
    ran, stopped = [], []
    done = threading.Event()

    def task(text):
        ran.append(text)
        done.set()

    def make(**kw):
        server = StreamServer(_Rig(), host="0.0.0.0", port=0, **kw)
        server.set_task_fn(task)
        server.set_cancel_fn(lambda: stopped.append(True))
        server.start()
        made.append(server)
        return server

    made: list[StreamServer] = []
    yield make, ran, stopped, done
    for s in made:
        s.stop()


def test_a_lan_browser_cannot_drive_the_arm_but_can_stop_it(lan_server):
    make, ran, stopped, _ = lan_server
    server = make()
    base = f"http://{_lan_ip()}:{server.port}"

    code, body = _post(f"{base}/task", {"task": "wave"})
    assert code == 403
    assert body["accepted"] is False
    assert "this machine" in body["error"]
    assert ran == [], "a refused command must never reach the reflex grammar"

    code, body = _post(f"{base}/cancel")
    assert code == 200 and body["cancelled"] is True
    assert stopped == [True], "STOP must keep working from any browser"


def test_lan_commands_opt_in_accepts_a_lan_browser(lan_server):
    make, ran, _, done = lan_server
    server = make(lan_commands=True)
    code, body = _post(f"http://{_lan_ip()}:{server.port}/task", {"task": "wave"})
    assert code == 200 and body["accepted"] is True
    assert done.wait(5) and ran == ["wave"]


def test_demo_config_keeps_lan_commands_off():
    from cascade.config import load_demo_config

    assert load_demo_config().get("stream").get("lan_commands") is False


@pytest.mark.parametrize("opt_in", [False, True])
def test_build_runtime_hands_the_config_key_to_the_live_view(tmp_path, monkeypatch, opt_in):
    """`stream.lan_commands` must reach the server the runtime actually binds."""
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config

    monkeypatch.setenv("CASCADE_STREAM", "eager")
    cfg = load_demo_config(camera="mock_small", arm="so101_mock", llm="mock")
    cfg._data["stream"]["port"] = 0
    cfg._data["stream"]["lan_commands"] = opt_in
    runtime, arm = build_runtime(cfg, tmp_path / "run", serve=True)
    try:
        server = runtime.live_view.server
        assert server is not None, "eager mode must have bound the live view"
        ran = []
        server.set_task_fn(ran.append)
        code, _ = _post(f"http://{_lan_ip()}:{server.port}/task", {"task": "wave"})
        assert code == (200 if opt_in else 403)
    finally:
        shutdown_runtime(runtime, arm)
