"""ASPIRE A10 (spec §4.8): the learning panel's data path, no new port.

The Spark camera surface (scripts/spark_cameras.py, its existing loopback port) adds the MCP server's
`<launch state>/learning.json` to its `/state` and `/api/status` replies as `learning`, only when the file is
schema 1 and at most 64 KB (`read_learning`; anything else: no key, as before); the launcher passes
`--learning-file <launch state>/learning.json`. The rig's own live view (apps/stream_server.py) adds the
runtime's learning state the same way, only for a runtime that built one (the kitchen): a hardware or default
/state has no `learning` key. The extension renders it as text (extensions/chrome/learning.mjs).

Every server here is an ephemeral loopback socket of the test's own: no live port is reached.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import socketserver
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import pytest
from conftest import needs_pin

ROOT = Path(__file__).resolve().parents[1]
SHA = "3fa9c2d1" + "0" * 56
LEARNING = {"schema": 1, "library_sha": SHA, "profile_class": "sim", "sim_retry": True,
            "served": [{"id": "red-cube", "title": "Cube near the lemon",
                        "tag": "learned in sim | verifier 4/5 | lib 3fa9c2d1"}],
            "counts": {"promoted": 1, "verified": 1, "candidates": 0, "refuted": 0},
            "last_event": {"t": 1.5, "kind": "note", "line": "From earlier runs: Cube near the lemon (x).",
                           "outcome": "placed"}}


def _cameras():
    spec = importlib.util.spec_from_file_location("spark_cameras_learning", ROOT / "scripts" / "spark_cameras.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── read_learning ────────────────────────────────────────────────────────


def test_read_learning_returns_a_valid_schema_1_file(tmp_path):
    cameras = _cameras()
    path = tmp_path / "learning.json"
    path.write_text(json.dumps(LEARNING))
    assert cameras.read_learning(path) == LEARNING
    assert cameras.LEARNING_MAX_BYTES == 64 * 1024


@pytest.mark.parametrize("content", [
    b"", b"{", b"not json", b"[]", b'"x"', b"1", b"null",
    json.dumps({**LEARNING, "schema": 2}).encode(),
    json.dumps({**LEARNING, "schema": "1"}).encode(),
    json.dumps({**LEARNING, "schema": True}).encode(),
    json.dumps({**LEARNING, "schema": 1.0}).encode(),
    json.dumps({k: v for k, v in LEARNING.items() if k != "schema"}).encode(),
    b"\xff\xfe" + json.dumps(LEARNING).encode("utf-16-le"),
], ids=["empty", "truncated", "text", "list", "string", "number", "null", "schema 2", "schema str",
        "schema bool", "schema float", "no schema", "utf-16"])
def test_read_learning_refuses_anything_else(tmp_path, content):
    cameras = _cameras()
    path = tmp_path / "learning.json"
    path.write_bytes(content)
    assert cameras.read_learning(path) is None


@pytest.mark.parametrize("content", [
    b"[" * 65000,
    b'{"schema": 1, "deep": ' + b"[" * 30000 + b"]" * 30000 + b"}",
    json.dumps({**LEARNING, "counts": {"promoted": float("nan")}}).encode(),
    json.dumps({**LEARNING, "t": float("inf")}).encode(),
    json.dumps({**LEARNING, "t": float("-inf")}).encode(),
], ids=["deep unclosed", "deep closed", "NaN", "Infinity", "-Infinity"])
def test_read_learning_refuses_what_a_browser_cannot_parse(tmp_path, content):  # A9/A10 fix, Minor 3
    """Deep nesting (Python's RecursionError, not a ValueError) and NaN / Infinity (Python accepts them, the
    extension's JSON.parse then fails the whole /state): no `learning` key, never an exception."""
    cameras = _cameras()
    path = tmp_path / "learning.json"
    path.write_bytes(content)
    assert len(content) <= cameras.LEARNING_MAX_BYTES
    assert cameras.read_learning(path) is None


def test_read_learning_never_blocks_on_a_fifo(tmp_path):  # A9/A10 fix, Minor 3
    """A FIFO at the path (no writer) would block the /state handler thread on open(): only a regular file is read."""
    import os

    cameras = _cameras()
    path = tmp_path / "learning.json"
    os.mkfifo(path)
    out = []
    reader = threading.Thread(target=lambda: out.append(cameras.read_learning(path)), daemon=True)
    reader.start()
    reader.join(5.0)
    if reader.is_alive():  # unblock the reader before failing, so no thread is left behind
        with open(path, "wb"):
            pass
        reader.join(5.0)
        pytest.fail("read_learning blocked on a FIFO")
    assert out == [None]


def test_read_learning_reads_only_a_regular_file(tmp_path):  # A9/A10 fix, Minor 3
    """A FIFO whose writer already wrote a valid schema-1 file is still no panel: only a regular file is read."""
    import os

    cameras = _cameras()
    path = tmp_path / "learning.json"
    os.mkfifo(path)
    keep = os.open(path, os.O_RDONLY | os.O_NONBLOCK)  # a reader, so the writer's open never blocks
    writer = os.open(path, os.O_WRONLY)
    try:
        os.write(writer, json.dumps(LEARNING).encode())
        assert cameras.read_learning(path) is None
    finally:
        os.close(writer)
        os.close(keep)


def test_read_learning_refuses_a_missing_path_a_directory_and_no_flag(tmp_path):
    cameras = _cameras()
    assert cameras.read_learning(None) is None
    assert cameras.read_learning(tmp_path / "missing.json") is None
    assert cameras.read_learning(tmp_path) is None


def test_read_learning_refuses_more_than_64_kb(tmp_path):
    cameras = _cameras()
    path = tmp_path / "learning.json"
    body = json.dumps({**LEARNING, "pad": ""})
    pad = 64 * 1024 - len(body)
    path.write_text(json.dumps({**LEARNING, "pad": "x" * pad}))
    assert path.stat().st_size == 64 * 1024 and cameras.read_learning(path) is not None
    path.write_text(json.dumps({**LEARNING, "pad": "x" * (pad + 1)}))
    assert cameras.read_learning(path) is None


# ── the camera surface's /state ──────────────────────────────────────────


class _Bridge(socketserver.StreamRequestHandler):
    def handle(self):
        json.loads(self.rfile.readline())
        jpg = b"\xff\xd8learning\xff\xd9"
        self.wfile.write(json.dumps({"ok": True, "t": 1.0, "rgb_jpeg_b64": base64.b64encode(jpg).decode()})
                         .encode() + b"\n")


@pytest.fixture
def surface(tmp_path):
    """The camera surface on an ephemeral loopback port with a fake bridge; `learning_file` as given."""
    cameras = _cameras()
    servers = []

    def start(learning_file=None, *, set_attr=True):
        bridge = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Bridge)
        bridge.daemon_threads = True
        http = ThreadingHTTPServer(("127.0.0.1", 0), cameras.CameraHandler)
        http.daemon_threads = True
        http.frames = cameras.Frames(bridge.server_address[1])
        if set_attr:
            http.learning_file = learning_file
        for server in (bridge, http):
            servers.append(server)
            threading.Thread(target=server.serve_forever, daemon=True).start()
        origin = f"http://127.0.0.1:{http.server_address[1]}"
        return lambda path: json.load(urlopen(origin + path, timeout=5))

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def test_state_includes_learning_only_from_a_valid_file(tmp_path, surface):
    path = tmp_path / "learning.json"
    get = surface(path)
    for route in ("/state", "/api/status"):
        assert "learning" not in get(route), "missing file: no key"
    path.write_text(json.dumps(LEARNING))
    for route in ("/state", "/api/status"):
        state = get(route)
        assert state["learning"] == LEARNING and set(state["cameras"]) == {"kitchen", "worktop", "side"}
    path.write_text(json.dumps({**LEARNING, "pad": "x" * (64 * 1024)}))
    assert "learning" not in get("/state"), "oversize: no key"
    path.write_text("{bad json")
    assert "learning" not in get("/state"), "bad JSON: no key"
    path.write_text(json.dumps({**LEARNING, "schema": 2}))
    assert "learning" not in get("/state"), "another schema: no key"


def test_state_without_the_flag_is_main_s(surface):
    for get in (surface(None), surface(set_attr=False)):
        state = get("/state")
        assert set(state) == {"cameras", "camera_discovery"}


def test_the_camera_surface_parses_the_learning_file_flag(tmp_path, monkeypatch):
    cameras = _cameras()
    made = []

    class Server:
        def __init__(self, address, handler):
            self.address = address
            made.append(self)

        def serve_forever(self):
            return None

    monkeypatch.setattr(cameras, "ThreadingHTTPServer", Server)
    for argv, expected in ((["spark_cameras.py", "--port", "0"], None),
                           (["spark_cameras.py", "--port", "0", "--learning-file", str(tmp_path / "l.json")],
                            tmp_path / "l.json")):
        monkeypatch.setattr(sys, "argv", argv)
        cameras.main()
        assert made[-1].learning_file == expected
        assert made[-1].address == ("127.0.0.1", 0)


# ── the launcher passes the flag ─────────────────────────────────────────


def test_camera_surface_gets_the_learning_file_flag(tmp_path, monkeypatch):
    """start_surfaces gives the camera surface `--learning-file <launch state>/learning.json` (the MCP
    server's launch state: the same `profile_state_dir(..., "cascade-demo")`), and nothing else changes."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import spark_browser

        import cascade.apps.process_owner as owner_module

        state = tmp_path / "state"
        state.mkdir()
        commands = []
        monkeypatch.setattr(spark_browser, "ownership", lambda repo: (state, {"owner": 1}))
        monkeypatch.setattr(owner_module, "live_records", lambda *a, **k: [])
        monkeypatch.setattr(owner_module, "register_process", lambda *a, **k: None)

        def refused(*_a, **_k):  # never reach a live port
            raise ConnectionRefusedError
        monkeypatch.setattr(spark_browser.socket, "create_connection", refused)
        monkeypatch.setattr(spark_browser, "_spawn_surface",
                            lambda command, *a, **k: commands.append(command) or SimpleNamespace(pid=1,
                                                                                                poll=lambda: None))
        monkeypatch.setattr(spark_browser, "ready", lambda repo, **k: True)
        spark_browser.start_surfaces(tmp_path, {})
        camera, visitor = commands
        assert camera == ["/usr/bin/python3", str(tmp_path.resolve() / "scripts/spark_cameras.py"),
                          "--learning-file", str(state / "learning.json")]
        assert visitor[1].endswith("deploy/brev/visitor.py") and "--learning-file" not in visitor
    finally:
        sys.path.remove(str(ROOT / "scripts"))
        sys.modules.pop("spark_browser", None)


def test_no_new_port():
    """The panel rides the camera surface's existing /state: no new listener, no new route."""
    source = (ROOT / "scripts" / "spark_cameras.py").read_text(encoding="utf-8")
    assert source.count("ThreadingHTTPServer((") == 1 and "socket.socket(" not in source
    assert '"/learning' not in source


# ── the rig's own live view (apps/stream_server.py) ──────────────────────


def _stream_state(runtime):
    from cascade.apps.stream_server import StreamServer

    server = StreamServer(SimpleNamespace(stats=lambda: {}), port=0, runtime_fn=lambda: runtime,
                          state_fn=lambda: {"agent_status": "idle"})
    return server.state()


def test_live_view_state_has_learning_only_from_a_runtime_that_built_it(tmp_path):
    from test_aspire_notes import _note

    from cascade.agent.learning_surface import LearningState
    from cascade.skills.notes import load_snapshot

    _note(tmp_path / "library", "red-cube", when="red cube")
    learning = LearningState([], load_snapshot(tmp_path / "library", "sim"), True)
    state = _stream_state(SimpleNamespace(learning_state=learning))
    assert state["learning"] == learning.payload() and state["learning"]["schema"] == 1
    for runtime in (None, SimpleNamespace(), SimpleNamespace(learning_state=None)):
        assert "learning" not in _stream_state(runtime)


@needs_pin
@pytest.mark.parametrize("surface", [True, False], ids=["kitchen", "surface off"])
def test_the_kitchen_live_view_serves_the_runtime_learning_json(monkeypatch, tmp_path, surface):
    """End to end on the mock kitchen: after a granted pick the live view's /state carries exactly the
    run directory's learning.json; with the switch off it has no key and no file exists."""
    from test_learning_surface import PICK, _kitchen

    with _kitchen(monkeypatch, tmp_path, gate=False, surface=surface) as (runtime, arm):
        assert runtime.execute("pick_and_place", PICK)["ok"] is True
        state = _stream_state(runtime)
        if surface:
            written = json.loads((tmp_path / "run" / "learning.json").read_text())
            assert state["learning"] == written and written["last_event"]["kind"] == "note"
        else:
            assert "learning" not in state and not (tmp_path / "run" / "learning.json").exists()


def test_live_view_state_survives_a_broken_learning_state():
    class Broken:
        def payload(self):
            raise RuntimeError("boom")

    state = _stream_state(SimpleNamespace(learning_state=Broken()))
    assert "learning" not in state and state["agent_status"] == "idle"


def test_live_view_state_never_serves_an_oversize_payload():
    class Huge:
        def payload(self):
            return {"schema": 1, "pad": "x" * (64 * 1024)}

    assert "learning" not in _stream_state(SimpleNamespace(learning_state=Huge()))
