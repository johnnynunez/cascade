"""B64: test servers bind a port the OS assigns and prove they are the test's own.

Measured on 2026-10-09 with several worktree full suites running at once on one
host: tests/test_graspgenx_backend.py started its protocol stub on a hard-coded
5599, tests/test_occupancy.py its bridge on 5598 / 5599, and each fixture
declared its server ready as soon as ANY listener accepted a TCP connection on
that port. The second suite's server could not bind, its fixture found the first
suite's server instead, and its tests talked to it: 2 graspgenx failures
(`zmq.error.Again`) and 4 occupancy failures (`KeyError: 'point_cloud'`).

Premises (pass on main by design): the wire clients accept ANY server that
answers their protocol, so identity has to be the fixture's own check; and a
server started on a port another socket holds cannot bind it -- on main the
second suite's stub died while its fixture carried on with the first's.

RED (fail on main, pass after): each fixture module runs in a child pytest with
the port its fixture is configured to use (on main: the module constant)
pointed at a foreign listener this test owns -- an impostor that answers every
request with an error and records it. The impostor sits on an OS-assigned port
held by this process, never on the literal 5598 / 5599: seizing those would
break any concurrent suite still on main, i.e. reproduce the bug on a sibling.
On main the fixture talks to the impostor; after B64 it never does.

The rest pins the mechanism (tests/owned_server.py): `--port 0` + one
`CASCADE_SERVER_READY` stdout line + `pid` / `instance` echoed over the real
protocol, refusal of every mismatch, and unchanged replies without
`--instance-id`.
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

import numpy as np
import pytest
from conftest import REPO, loopback_host
from owned_server import (OwnedServerError, held_dead_port, spawn_announced,
                          start_owned_server)

STUB = REPO / "scripts" / "serve_graspgenx_stub.py"
BRIDGE = REPO / "scripts" / "serve_occupancy_bridge.py"
ENDPOINT_VARIABLES = ("CASCADE_GRASPGENX_PORT", "CASCADE_OCCUPANCY_PORT", "CASCADE_BRIDGE_PORT",
                      "CASCADE_HUG_PORT", "CASCADE_GRASPGENX_HOST", "CASCADE_HUG_HOST")
OLD_PROBE_KEYS = {"ok", "backend", "describe", "voxel", "device", "esdf", "carving", "masked_depth"}


def _has_wire() -> bool:
    try:
        import msgpack  # noqa: F401
        import msgpack_numpy  # noqa: F401
        import zmq  # noqa: F401

        return True
    except ImportError:
        return False


needs_wire = pytest.mark.skipif(
    not _has_wire(), reason="needs the grasping extra: uv pip install -e '.[grasping]'")


class _Impostor:
    """A foreign ZMQ server on `count` consecutive OS-assigned loopback ports
    held by THIS process. It answers every request with `reply` and records the
    action, so a test can tell whether anyone talked to it."""

    def __init__(self, count: int = 1, reply: dict | None = None):
        import zmq

        self._zmq = zmq
        self._ctx = zmq.Context()
        self._reply = reply or {"error": "impostor: a foreign server on this fixture's port"}
        self.requests: list = []
        self._sockets = self._bind(count)
        self.ports = [self._port_of(sock) for sock in self._sockets]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="b64-impostor", daemon=True)
        self._thread.start()

    def _port_of(self, sock) -> int:
        return int(sock.getsockopt(self._zmq.LAST_ENDPOINT).decode().rsplit(":", 1)[1])

    def _socket(self):
        sock = self._ctx.socket(self._zmq.REP)
        sock.setsockopt(self._zmq.LINGER, 0)
        return sock

    def _bind(self, count: int) -> list:
        for _ in range(100):
            held = [self._socket()]
            held[0].bind("tcp://127.0.0.1:0")
            base = self._port_of(held[0])
            try:
                for offset in range(1, count):
                    held.append(self._socket())
                    held[-1].bind(f"tcp://127.0.0.1:{base + offset}")
                return held
            except self._zmq.ZMQError:
                for sock in held:
                    sock.close()
        raise RuntimeError(f"no run of {count} free consecutive loopback ports")

    def _serve(self) -> None:
        import msgpack

        poller = self._zmq.Poller()
        for sock in self._sockets:
            poller.register(sock, self._zmq.POLLIN)
        while not self._stop.is_set():
            for sock, _ in poller.poll(50):
                raw = sock.recv()
                try:
                    request = msgpack.unpackb(raw, raw=False, strict_map_key=False)
                    action = request.get("action") if isinstance(request, dict) else request
                except Exception:  # noqa: BLE001 -- recorded, never fatal
                    action = "<undecodable>"
                self.requests.append(action)
                sock.send(msgpack.packb(self._reply, use_bin_type=True))

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=30)
        for sock in self._sockets:
            sock.close()
        self._ctx.term()


@contextmanager
def _impostor(count: int = 1, reply: dict | None = None):
    impostor = _Impostor(count, reply)
    try:
        yield impostor
    finally:
        impostor.close()


_FIXTURE_PORTS_PLUGIN = '''
import json, os, sys


def pytest_collection_finish(session):
    for module, values in json.loads(os.environ["B64_FIXTURE_PORTS"]).items():
        for name, value in values.items():
            setattr(sys.modules[module], name, value)
'''


def _run_fixture_module(tmp_path, module: str, select: str, ports: dict) -> tuple[int, str]:
    """Run `select` from tests/<module>.py in a child pytest after setting the
    module's fixture port constants (main's only port mechanism) to `ports`."""
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "b64_fixture_ports.py").write_text(_FIXTURE_PORTS_PLUGIN)
    env = {k: v for k, v in os.environ.items() if k not in ENDPOINT_VARIABLES}
    env["B64_FIXTURE_PORTS"] = json.dumps({module: ports})
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(plugin_dir), str(REPO / "src"), os.environ.get("PYTHONPATH")) if p)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
         "-p", "b64_fixture_ports", f"--basetemp={tmp_path / 'child'}", "-k", select,
         str(REPO / "tests" / f"{module}.py")],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=1200)
    return result.returncode, result.stdout + result.stderr


def _stub_identity(port: int) -> dict:
    from cascade.grasping.graspgenx_backend import GraspGenXClient

    client = GraspGenXClient(host="127.0.0.1", port=port, timeout_ms=30000)
    try:
        return client.probe(timeout_ms=30000)
    finally:
        client.close()


def _bridge_identity(port: int) -> dict:
    from cascade.perception.occupancy import OccupancyClient

    client = OccupancyClient(host=loopback_host(), port=port, timeout_ms=30000)
    try:
        return client.probe(timeout_ms=30000)
    finally:
        client.close()


def _cube_fix():
    from cascade.types import Detection, ObjectFix

    rng = np.random.default_rng(0)
    centre = np.array([0.22, 0.0, 0.045])
    return ObjectFix(label="red cube", position=centre,
                     points=centre + rng.uniform(-0.025, 0.025, size=(400, 3)),
                     detection=Detection(label="red cube", conf=0.9, bbox=np.zeros(4)),
                     extent=np.array([0.05] * 3), axes=np.eye(3))


# ── premises: pass on main by design ─────────────────────────────────────


@needs_wire
def test_premise_the_wire_clients_accept_any_server_that_answers_their_protocol():
    """The clients carry no notion of WHICH server they expect (a launcher may
    restart a sidecar), so a fixture that finds "a server" on its port has
    proven nothing: identity must be the fixture's own check."""
    from cascade.grasping.graspgenx_backend import GraspGenXClient
    from cascade.perception.occupancy import OccupancyClient

    with _impostor(reply={"status": "ok", "ok": True, "stub": True}) as foreign:
        client = GraspGenXClient(host="127.0.0.1", port=foreign.ports[0], timeout_ms=30000)
        try:
            assert client.probe(timeout_ms=30000)["stub"] is True
        finally:
            client.close()
        assert foreign.requests == ["health"]
    with _impostor(reply={"ok": True, "backend": "warp", "esdf": True}) as foreign:
        client = OccupancyClient(host="127.0.0.1", port=foreign.ports[0], timeout_ms=30000)
        try:
            assert client.probe(timeout_ms=30000)["backend"] == "warp"
        finally:
            client.close()
        assert foreign.requests == ["probe"]


@needs_wire
def test_premise_a_stub_cannot_bind_a_port_another_socket_holds():
    """Why the second suite's stub died on main: its bind failed while the first
    suite's stub held 5599 (here: a port this test holds), and the old fixture
    still found "a listener" there and carried on."""
    with held_dead_port() as port:
        result = subprocess.run([sys.executable, str(STUB), "--quiet", "--port", str(port)],
                                capture_output=True, text=True, timeout=300)
    assert result.returncode != 0
    assert "Address already in use" in result.stderr


def _connect_outcome(port: int, timeout_s: float) -> str:
    """'connected', 'refused' or 'no answer' for one TCP connect to `port`."""
    try:
        socket.create_connection(("127.0.0.1", port), timeout=timeout_s).close()
    except ConnectionRefusedError:
        return "refused"
    except TimeoutError:
        return "no answer"
    return "connected"


def test_a_held_dead_port_never_completes_a_connection_and_cannot_be_taken():
    """The contract every `dead_port` fixture relies on, on every platform: no
    connection to the port ever completes, and no other socket can bind it.
    HOW the connect fails is platform-specific (next test): the macOS CI runner
    never answers it, so this probe gives up after 2 s instead of 30."""
    with held_dead_port() as port:
        assert _connect_outcome(port, timeout_s=2.0) in ("refused", "no answer")
        intruder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(OSError):
                intruder.bind(("127.0.0.1", port))
        finally:
            intruder.close()


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="Linux-only: macOS leaves a connect to a bound, non-listening port unanswered")
def test_on_linux_a_held_dead_port_refuses_at_once():
    """Linux answers a connect to a bound, non-listening port with a reset, so a
    client fails immediately. Measured on the macOS CI runner (main 68d2866,
    the first version of this test): the same connect timed out after 30 s
    (`TimeoutError: [Errno 60]`), i.e. a dead_port client there fails by its
    own timeout instead -- why every dead_port test uses a short client
    timeout."""
    with held_dead_port() as port:
        t0 = time.monotonic()
        assert _connect_outcome(port, timeout_s=30.0) == "refused"
        assert time.monotonic() - t0 < 1.0


# ── RED: fail on main, pass after ────────────────────────────────────────


@needs_wire
def test_the_stub_fixture_never_talks_to_a_foreign_listener_on_its_port(tmp_path):
    with _impostor() as foreign:
        code, out = _run_fixture_module(
            tmp_path, "test_graspgenx_backend",
            "test_grasps_come_back_and_are_top_down or test_grasps_are_ranked_best_first",
            {"PORT": foreign.ports[0]})
    assert foreign.requests == [], (
        f"the stub fixture's tests talked to a foreign listener: {foreign.requests}\n{out[-2500:]}")
    assert code == 0 and re.search(r"\b2 passed\b", out), out[-2500:]


@needs_wire
def test_the_bridge_fixtures_never_talk_to_a_foreign_listener_on_their_ports(tmp_path):
    pytest.importorskip("warp")
    with _impostor(count=2) as foreign:     # main: bridge_server on N, fresh_bridge on N + 1
        code, out = _run_fixture_module(
            tmp_path, "test_occupancy",
            "test_the_bridge_reports_a_bad_action_as_an_error or test_a_real_depth_frame_becomes_clearance",
            {"BRIDGE_PORT": foreign.ports[0]})
    assert foreign.requests == [], (
        f"the bridge fixtures' tests talked to a foreign listener: {foreign.requests}\n{out[-2500:]}")
    assert code == 0 and re.search(r"\b2 passed\b", out), out[-2500:]


@needs_wire
def test_the_stub_on_an_os_assigned_port_names_itself_and_plans_grasps(tmp_path):
    from cascade.config import Cfg
    from cascade.grasping.graspgenx_backend import GraspGenXPlanner

    server = start_owned_server([sys.executable, str(STUB), "--quiet"], identify=_stub_identity,
                                deadline_s=120.0, stderr_path=tmp_path / "stub.stderr")
    try:
        reply = _stub_identity(server.port)
        assert (reply["pid"], reply["instance"], reply["stub"]) == (server.proc.pid, server.instance, True)
        planner = GraspGenXPlanner(Cfg({"graspgenx": {
            "host": "127.0.0.1", "port": server.port, "timeout_ms": 30000, "num_grasps": 8,
            "tip_offset_m": 0.098}}))
        assert planner.plan(_cube_fix(), max_width_m=0.055), "no grasps over the owned port"
    finally:
        assert server.stop(), "the stub exited while the test used it"


@needs_wire
def test_the_bridge_on_an_os_assigned_port_names_itself_and_serves_the_map(tmp_path):
    from cascade.perception.occupancy import OccupancyClient

    server = start_owned_server(
        [sys.executable, str(BRIDGE), "--backend", "voxel", "--voxel-size", "0.02"],
        identify=_bridge_identity, deadline_s=180.0, stderr_path=tmp_path / "bridge.stderr")
    try:
        reply = _bridge_identity(server.port)
        assert (reply["pid"], reply["instance"], reply["backend"]) == (
            server.proc.pid, server.instance, "voxel")
        client = OccupancyClient(host=loopback_host(), port=server.port, timeout_ms=30000)
        try:
            wall = np.array([[0.30, y, 0.15] for y in np.linspace(-0.1, 0.1, 50)], dtype=np.float32)
            assert client.request({"action": "integrate", "points": wall}) == {}
            resp = client.request({"action": "query",
                                   "region_min": np.array([0.0, -0.3, 0.0], dtype=np.float32),
                                   "region_max": np.array([0.6, 0.3, 0.4], dtype=np.float32)})
            assert len(np.asarray(resp["points"]).reshape(-1, 3)) > 0
        finally:
            client.close()
    finally:
        assert server.stop(), "the bridge exited while the test used it"


# ── golden: without --instance-id the replies are byte-for-byte the old ones ──


@needs_wire
def test_without_an_instance_id_the_stub_health_reply_is_unchanged(tmp_path):
    from cascade.grasping.graspgenx_backend import GraspGenXClient

    proc, ready = spawn_announced([sys.executable, str(STUB), "--quiet"], deadline_s=120.0,
                                  stderr_path=tmp_path / "stub.stderr")
    try:
        assert (ready["pid"], ready["instance"]) == (proc.pid, None)
        assert ready["endpoint"] == f"tcp://127.0.0.1:{ready['port']}"
        client = GraspGenXClient(host="127.0.0.1", port=ready["port"], timeout_ms=30000)
        try:
            for action in ("health", "ping"):
                assert client.request({"action": action}) == {"status": "ok", "ok": True, "stub": True}
        finally:
            client.close()
    finally:
        proc.kill()
        proc.wait(timeout=30)


@needs_wire
def test_without_an_instance_id_the_bridge_probe_reply_is_unchanged(tmp_path):
    from cascade.perception.occupancy import OccupancyClient

    proc, ready = spawn_announced([sys.executable, str(BRIDGE), "--backend", "voxel"],
                                  deadline_s=180.0, stderr_path=tmp_path / "bridge.stderr")
    try:
        assert (ready["pid"], ready["instance"]) == (proc.pid, None)
        client = OccupancyClient(host=loopback_host(), port=ready["port"], timeout_ms=30000)
        try:
            assert set(client.probe(timeout_ms=30000)) == OLD_PROBE_KEYS
        finally:
            client.close()
    finally:
        proc.kill()
        proc.wait(timeout=30)
    assert f":{ready['port']} backend=voxel" in (tmp_path / "bridge.stderr").read_text()


@needs_wire
def test_two_owned_stubs_run_side_by_side_and_each_answers_as_itself(tmp_path):
    """Two suites at once: each fixture gets its own port and its own server."""
    argv = [sys.executable, str(STUB), "--quiet"]
    first = start_owned_server(argv, identify=_stub_identity, deadline_s=120.0,
                               stderr_path=tmp_path / "first.stderr")
    try:
        second = start_owned_server(argv, identify=_stub_identity, deadline_s=120.0,
                                    stderr_path=tmp_path / "second.stderr")
        try:
            assert first.port != second.port and first.instance != second.instance
            assert _stub_identity(first.port)["instance"] == first.instance
            assert _stub_identity(second.port)["instance"] == second.instance
        finally:
            assert second.stop()
    finally:
        assert first.stop()


# ── the acceptance rule itself, on a stdlib fake server (no wire extra) ──


_FAKE_SERVER = r'''
import json, os, socket, sys, time

args = sys.argv[1:]
mode = args[args.index("--mode") + 1]
if args[args.index("--port") + 1] != "0":
    sys.exit("fake server: the launcher did not ask for an OS-assigned port")
instance = args[args.index("--instance-id") + 1]
pid = os.getpid()
if mode == "exit":
    sys.stderr.write("fake server: cannot start (boom)\n")
    sys.exit(3)
if mode == "silent":
    time.sleep(600)
listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen()
port = listener.getsockname()[1]
ready = {"endpoint": "tcp://127.0.0.1:%d" % port, "port": port, "pid": pid, "instance": instance}
reply = {"ok": True, "pid": pid, "instance": instance}
if mode == "noise":
    print("Warp 1.0 initialized:", flush=True)
    print("CASCADE_SERVER_READ", flush=True)
elif mode == "ready-pid":
    ready["pid"] = pid + 1
elif mode == "ready-instance":
    ready["instance"] = "someone-else"
elif mode.startswith("ready-port="):
    ready["port"] = json.loads(mode.split("=", 1)[1])
elif mode == "reply-pid":
    reply["pid"] = pid + 1
elif mode == "reply-instance":
    reply["instance"] = "someone-else"
elif mode.startswith("relay="):
    ready["port"] = int(mode.split("=", 1)[1])
print("CASCADE_SERVER_READY " + json.dumps(ready), flush=True)
while True:
    connection, _ = listener.accept()
    with connection:
        connection.sendall((json.dumps(reply) + "\n").encode())
'''


def _line_identity(port: int) -> dict:
    with socket.create_connection(("127.0.0.1", port), timeout=30) as connection:
        return json.loads(connection.makefile().readline())


@pytest.fixture
def fake(tmp_path):
    script = tmp_path / "fake_server.py"
    script.write_text(_FAKE_SERVER)

    def start(mode: str, deadline_s: float = 120.0):
        return start_owned_server([sys.executable, str(script), "--mode", mode],
                                  identify=_line_identity, deadline_s=deadline_s,
                                  stderr_path=tmp_path / f"fake-{len(list(tmp_path.iterdir()))}.stderr")
    return start


@contextmanager
def _line_server(reply: dict):
    """A foreign line server on an OS-assigned port, owned by this test."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()

    def serve():
        while True:
            try:
                connection, _ = listener.accept()
            except OSError:
                return
            with connection:
                connection.sendall((json.dumps(reply) + "\n").encode())

    threading.Thread(target=serve, daemon=True).start()
    try:
        yield listener.getsockname()[1]
    finally:
        listener.close()


@pytest.mark.parametrize("mode", ["honest", "noise"])
def test_a_server_naming_its_pid_and_token_is_accepted(fake, mode):
    server = fake(mode)
    reply = _line_identity(server.port)
    assert (reply["pid"], reply["instance"]) == (server.proc.pid, server.instance)
    assert len(server.instance) == 16
    assert server.stop() is True
    assert server.proc.returncode == -signal.SIGKILL


def test_stop_reports_a_server_that_died_while_the_tests_ran(fake):
    server = fake("honest")
    server.proc.kill()
    server.proc.wait(timeout=30)
    assert server.stop() is False


@pytest.mark.parametrize("mode, words", [
    ("ready-pid", "ready line names pid"),
    ("ready-instance", "ready line names instance"),
    ("reply-pid", "protocol reply names pid"),
    ("reply-instance", "protocol reply names instance"),
    ("ready-port=0", "no usable port"),
    ("ready-port=70000", "no usable port"),
    ('ready-port="5599"', "no usable port"),
    ("ready-port=true", "no usable port"),
])
def test_a_server_that_names_someone_else_is_refused_and_killed(fake, mode, words):
    with pytest.raises(OwnedServerError, match=words) as info:
        fake(mode)
    assert info.value.proc.returncode == -signal.SIGKILL


def test_a_foreign_listener_behind_a_correct_ready_line_is_refused(fake):
    """The ready line can be right while the port answers for someone else."""
    with _line_server({"ok": True, "pid": os.getpid(), "instance": "impostor"}) as port:
        with pytest.raises(OwnedServerError, match="protocol reply names pid") as info:
            fake(f"relay={port}")
    assert info.value.proc.returncode == -signal.SIGKILL


def test_a_failed_identity_round_trip_is_refused_and_killed(fake):
    with held_dead_port() as dead:
        with pytest.raises(OwnedServerError, match="identity round trip") as info:
            fake(f"relay={dead}")
    assert info.value.proc.returncode == -signal.SIGKILL


def test_a_server_that_exits_before_announcing_reports_its_stderr(fake):
    with pytest.raises(OwnedServerError, match=r"exited \(3\) before its ready line.*boom") as info:
        fake("exit")
    assert info.value.proc.returncode == 3


def test_a_server_that_never_announces_is_killed_at_the_deadline(fake):
    with pytest.raises(OwnedServerError, match="printed no CASCADE_SERVER_READY line in time") as info:
        fake("silent", deadline_s=1.0)
    assert info.value.proc.returncode == -signal.SIGKILL
