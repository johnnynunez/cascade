"""B70: the launcher checks the GraspGen-X endpoint the runtime will dial, and no
test takes a port by binding 0 and releasing it before its server uses it.

(a) `load_demo_config` applies `CASCADE_GRASPGENX_HOST` last over
`grasp.graspgenx.host` (B41), so with the variable set the runtime dials that
host. `scripts/launch.sh --graspgenx external` still checked
`127.0.0.1:$CASCADE_GRASPGENX_PORT`: a server on the other machine failed the
launch, and a loopback server the runtime would never use passed it. Its
learned-inference check (`check_graspgenx.py`, also run for `local`) never got
the host either. The launcher's own block is run here (the GraspGen-X part of
section 3b, `port_open`, `log`/`warn`/`die` and the banner line, cut from
scripts/launch.sh, with `start_sidecar` and `check_graspgenx.py` replaced by
recorders) against plain listeners this test owns on OS-assigned ports. A
listener on a second loopback address (127.0.0.2) stands for "another host";
where the platform has no such address (macOS configures only 127.0.0.1) those
tests skip and the rest still run.

Premises (pass on main by design): the runtime resolves the host variable;
`check_graspgenx.py --host` reaches its client; launch.sh parses.
Golden pins (pass on main): unset or empty host, and the `stub` / `none` modes
whatever the variable says, keep the launcher's argv and output byte for byte.
RED (fail on main): the external check dials the runtime host, names it, and
fails when only loopback answers; `local` and `external` pass `--host` to the
readiness check; a malformed host is refused by the runtime's rule before
anything is dialled or started.

(b) tests/test_hug_backend.py `_free_port` and the tests/test_openclaw_gateway.py
fixture took a "free" port by binding 0 and releasing it before their dead
endpoint / fake gateway used it. The dead endpoint is now `held_dead_port()`;
the gateway, which must be TOLD its port, is handed the very socket bound to it
(`owned_server.handed_over_port`, SCM_RIGHTS), so the port is never free between
selection and use. An AST guard fails on any new bind-0-read-close-use site in
tests/; the four older sites it still finds are listed below with their reasons.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
from conftest import REPO
from owned_server import held_dead_port

LAUNCH = REPO / "scripts" / "launch.sh"
TESTS = REPO / "tests"
CHECKED = "[launch] grasp planner: GraspGen-X CUDA model; diffusion inference checked before robot startup"
SHELLS = ["bash"] + (["/bin/bash"] if Path("/bin/bash").exists() else [])  # /bin/bash is 3.2 on macOS CI


# ── (a) the launcher's GraspGen-X check ────────────────────────────────────


def _launch_block() -> str:
    """The GraspGen-X part of launch.sh section 3b plus what it calls, verbatim."""
    source = LAUNCH.read_text()
    helpers = [line for line in source.splitlines() if re.match(r"(log|warn|die)\(\)\s+\{", line)]
    assert len(helpers) == 3, helpers
    port_open = source[source.index("port_open() {"):source.index("wait_port() {")]
    begin = source.index('case "$GRASPGENX" in\n    auto)')
    end = source.index("# ── 4. openclaw", begin)
    [banner] = [line for line in source.splitlines() if line.startswith('$( [[ "$GRASPGENX" != "none" ]]')]
    return ("\n".join(helpers) + "\n" + port_open + source[begin:end]
            + "cat <<EOF\n" + banner + "\nEOF\n")


def _run_launcher(tmp_path: Path, mode: str, port: int, host: str | None = None, *,
                  shell: str = "bash") -> tuple[subprocess.CompletedProcess, list[list[str]]]:
    """Run the block for `--graspgenx <mode>`; (result, argv of every readiness check)."""
    repo, state = tmp_path / "repo", tmp_path / "state"
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    state.mkdir(exist_ok=True)
    calls = tmp_path / "check_calls.jsonl"
    (repo / "scripts" / "check_graspgenx.py").write_text(
        "import json, os, sys\n"
        "with open(os.environ['B70_CHECK_CALLS'], 'a') as out:\n"
        "    out.write(json.dumps(sys.argv[1:]) + '\\n')\n")
    settings = {"PY": sys.executable, "REPO": repo, "STATE_DIR": state, "DRY": 0, "SIM": "isaac",
                "GRASPGENX": mode, "GGX_PORT": port}
    script = tmp_path / "ggx_check.sh"
    script.write_text("set -euo pipefail\n"
                      + "".join(f"{key}={shlex.quote(str(value))}\n" for key, value in settings.items())
                      + "start_sidecar() { printf 'START %s\\n' \"$*\"; }\n" + _launch_block())
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", str(tmp_path)),
           "PYTHONPATH": str(REPO / "src"), "B70_CHECK_CALLS": str(calls),
           "CASCADE_GRASPGENX_PORT": str(port)}
    if host is not None:
        env["CASCADE_GRASPGENX_HOST"] = host
    result = subprocess.run([shell, str(script)], env=env, capture_output=True, text=True, timeout=60)
    recorded = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    return result, recorded


def _second_loopback() -> str:
    """127.0.0.2 when this host can bind it (Linux: all of 127/8), else skip."""
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.2", 0))
    except OSError as exc:
        pytest.skip(f"127.0.0.2 is not a local address here ({exc}); macOS configures only 127.0.0.1")
    finally:
        probe.close()
    return "127.0.0.2"


@contextmanager
def _listener(host: str):
    """A plain TCP listener this test owns, on a port the OS assigns."""
    server = socket.socket()
    try:
        server.bind((host, 0))
        server.listen(8)
        yield server
    finally:
        server.close()


def _accepted(server: socket.socket) -> int:
    """Connections that reached `server` (completed handshakes in its backlog)."""
    server.setblocking(False)
    count = 0
    while True:
        try:
            connection, _ = server.accept()
        except (BlockingIOError, InterruptedError):
            return count
        connection.close()
        count += 1


def _readiness(tmp_path: Path, port: int, host: str | None = None) -> list[str]:
    head = [] if host is None else ["--host", host]
    return head + ["--port", str(port), "--output", str(tmp_path / "state" / "graspgenx-readiness.json")]


def test_premise_the_runtime_dials_the_host_variable(monkeypatch):
    # The cause: with CASCADE_GRASPGENX_HOST set, GraspGenXPlanner's endpoint
    # (config.sidecar_endpoint over the resolved section) is that host.
    from cascade.config import GRASPGENX_HOST_ENV, GRASPGENX_PORT_ENV, load_demo_config, sidecar_endpoint

    with held_dead_port() as port:
        monkeypatch.setenv("CASCADE_GRASPGENX_PORT", str(port))
        for host, expected in ((None, "127.0.0.1"), ("", "127.0.0.1"), ("gx10", "gx10")):
            if host is None:
                monkeypatch.delenv("CASCADE_GRASPGENX_HOST", raising=False)
            else:
                monkeypatch.setenv("CASCADE_GRASPGENX_HOST", host)
            cfg = load_demo_config(cameras=["mock"], arm="mock", llm="mock")
            views = [cfg] + [arm["resolved"] for arm in (cfg.get("arms") or [])]
            assert len(views) == 2                 # the top level and the one arm's own view
            for view in views:
                endpoint = sidecar_endpoint(view.get("grasp").get("graspgenx"), port_env=GRASPGENX_PORT_ENV,
                                            host_env=GRASPGENX_HOST_ENV, default_port=5556)
                assert endpoint == (expected, port)


def test_premise_check_graspgenx_dials_its_host_argument(monkeypatch):
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import check_graspgenx
    finally:
        sys.path.remove(str(REPO / "scripts"))
    from cascade.grasping.graspgenx_backend import GraspGenXError

    dialled = []

    class Recorder:
        def __init__(self, host, port, timeout_ms):
            dialled.append((host, port))

        def probe(self, timeout_ms):
            raise GraspGenXError("recorder: no server")

        def close(self):
            pass

    monkeypatch.setattr(check_graspgenx, "GraspGenXClient", Recorder)
    for argv, expected in ((["--host", "gx10", "--port", "47201"], ("gx10", 47201)),
                           (["--port", "47201"], ("127.0.0.1", 47201))):
        monkeypatch.setattr(sys, "argv", ["check_graspgenx.py", *argv])
        with pytest.raises(GraspGenXError):
            check_graspgenx.main()
        assert dialled.pop() == expected


@pytest.mark.parametrize("shell", SHELLS)
def test_premise_launch_sh_parses(shell):
    result = subprocess.run([shell, "-n", str(LAUNCH)], capture_output=True, text=True, timeout=30)
    assert (result.returncode, result.stderr) == (0, "")


@pytest.mark.parametrize("host", [None, ""], ids=["unset", "empty"])
def test_golden_default_external_check_is_unchanged(tmp_path, host):
    with _listener("127.0.0.1") as server:
        port = server.getsockname()[1]
        result, checks = _run_launcher(tmp_path, "external", port, host)
        assert _accepted(server) == 1
    assert (result.returncode, result.stderr) == (0, "")
    assert result.stdout == f"{CHECKED}\n         graspgenx: :{port} (external)\n"
    assert checks == [_readiness(tmp_path, port)]


def test_golden_default_external_check_still_fails_when_nothing_answers(tmp_path):
    with held_dead_port() as port:
        result, checks = _run_launcher(tmp_path, "external", port)
    assert result.returncode == 1
    assert (result.stdout, result.stderr) == ("", f"[launch] ERROR: no GraspGen-X server on :{port}\n")
    assert checks == []


@pytest.mark.parametrize("mode", ["stub", "none"])
def test_golden_stub_and_none_never_read_the_host_variable(tmp_path, mode):
    with held_dead_port() as port:
        runs = {host: _run_launcher(tmp_path / f"run{i}", mode, port, host)
                for i, host in enumerate([None, "", "127.0.0.2", "gx10:5556"])}
    baseline, _ = runs[None]
    assert baseline.returncode == 0 and baseline.stderr == ""
    if mode == "stub":
        assert baseline.stdout == (
            f"START graspgenx_stub {port} 30 {sys.executable} {tmp_path / 'run0' / 'repo'}"
            f"/scripts/serve_graspgenx_stub.py --port {port} --quiet\n"
            "[launch] grasp planner: GraspGen-X PROTOCOL STUB (analytic protocol double)\n"
            f"         graspgenx: :{port} (stub)\n")
    else:
        assert baseline.stdout == (
            "[launch] GraspGen-X explicitly disabled (--graspgenx none): analytic OBB planner\n\n")
    for i, (host, (result, checks)) in enumerate(runs.items()):
        same = result.stdout.replace(str(tmp_path / f"run{i}"), str(tmp_path / "run0"))
        assert (result.returncode, same, result.stderr, checks) == (0, baseline.stdout, "", []), host


@pytest.mark.parametrize("shell", SHELLS)
def test_external_check_dials_the_host_the_runtime_dials(tmp_path, shell):
    host = _second_loopback()
    with _listener(host) as server:                # nothing listens on 127.0.0.1 for this port
        port = server.getsockname()[1]
        result, checks = _run_launcher(tmp_path, "external", port, host, shell=shell)
        assert _accepted(server) == 1              # port_open dialled the runtime's host
    assert (result.returncode, result.stderr) == (0, ""), result.stdout
    assert result.stdout.splitlines() == [
        f"[launch] GraspGen-X endpoint: {host}:{port} (CASCADE_GRASPGENX_HOST); the checks below dial it",
        CHECKED,
        f"         graspgenx: {host}:{port} (external)",
    ]
    assert checks == [_readiness(tmp_path, port, host)]


def test_external_check_fails_when_only_loopback_answers(tmp_path):
    # A loopback server the runtime would never use must not pass the check.
    host = _second_loopback()
    with _listener("127.0.0.1") as loopback:
        port = loopback.getsockname()[1]
        dead = socket.socket()                     # hold host:port so nothing can listen there
        try:
            try:
                dead.bind((host, port))
            except OSError as exc:
                pytest.skip(f"{host}:{port} is taken here ({exc})")
            result, checks = _run_launcher(tmp_path, "external", port, host)
        finally:
            dead.close()
        assert _accepted(loopback) == 0
    assert result.returncode == 1
    assert result.stderr == f"[launch] ERROR: no GraspGen-X server on {host}:{port}\n"
    assert checks == []


def test_local_readiness_check_dials_the_runtime_host(tmp_path):
    # `local` starts the server here (start_sidecar is a recorder) but the
    # learned-inference check must ask the endpoint the runtime will use.
    with held_dead_port() as port:
        result, checks = _run_launcher(tmp_path, "local", port, "gx10")
    assert (result.returncode, result.stderr) == (0, ""), result.stdout
    lines = result.stdout.splitlines()
    assert lines[0] == f"[launch] GraspGen-X endpoint: gx10:{port} (CASCADE_GRASPGENX_HOST); the checks below dial it"
    assert lines[1].startswith(f"START graspgenx {port} 180 bash ")
    assert lines[-1] == f"         graspgenx: gx10:{port} (local)"
    assert checks == [_readiness(tmp_path, port, "gx10")]


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("mode", ["external", "local"])
def test_a_malformed_host_is_refused_before_anything_is_dialled_or_started(tmp_path, mode, shell):
    with _listener("127.0.0.1") as loopback:
        port = loopback.getsockname()[1]
        result, checks = _run_launcher(tmp_path, mode, port, "gx10:5556", shell=shell)
        assert _accepted(loopback) == 0
    assert result.returncode == 1
    assert result.stderr.startswith("[launch] ERROR: CASCADE_GRASPGENX_HOST='gx10:5556' is not a host")
    assert "Traceback" not in result.stderr and "unbound variable" not in result.stderr
    assert result.stdout == "" and checks == []


# ── (b) no port is released between selection and use ─────────────────────


def _scopes(tree: ast.AST) -> list[tuple[str, list[ast.AST]]]:
    """(name, nodes) of the module and of every function/class, nested scopes apart."""
    found: list[tuple[str, list[ast.AST]]] = []

    def collect(owner: ast.AST, name: str) -> None:
        nodes, stack = [], list(ast.iter_child_nodes(owner))
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                collect(node, node.name)
                continue
            nodes.append(node)
            stack.extend(ast.iter_child_nodes(node))
        found.append((name, nodes))

    collect(tree, "<module>")
    return found


def _receiver(node: ast.AST, method: str) -> str | None:
    """`x` for a call `x.<method>(...)` on a plain name."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == method and isinstance(node.func.value, ast.Name)):
        return node.func.value.id
    return None


def _released_port_sites(source: str) -> set[str]:
    """Scopes that bind a socket to port 0, read the number, release the socket
    (`close()` or the end of its `with` block) without ever listening on it,
    and use the number afterwards: the port is free again before it is used."""
    sites = set()
    for scope, nodes in _scopes(ast.parse(source)):
        bound_zero, listening, released = set(), set(), {}
        for node in nodes:
            name = _receiver(node, "bind")
            if (name and node.args and isinstance(node.args[0], ast.Tuple) and len(node.args[0].elts) == 2
                    and isinstance(node.args[0].elts[1], ast.Constant) and node.args[0].elts[1].value == 0):
                bound_zero.add(name)
            if _receiver(node, "listen"):
                listening.add(_receiver(node, "listen"))
            if _receiver(node, "close"):
                released.setdefault(_receiver(node, "close"), []).append(node.lineno)
            if isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if isinstance(item.optional_vars, ast.Name):
                        released.setdefault(item.optional_vars.id, []).append(node.end_lineno)
        for sock in bound_zero - listening:
            numbers = set()
            for node in nodes:
                if isinstance(node, ast.Assign) and any(_receiver(call, "getsockname") == sock
                                                        for call in ast.walk(node.value)):
                    numbers |= {t.id for target in node.targets for t in ast.walk(target)
                                if isinstance(t, ast.Name) and t.id != "_"}
            if any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in numbers
                   and node.lineno > line for node in nodes for line in released.get(sock, [])):
                sites.add(scope)
    return sites


#: Older sites the guard still finds. Not the race B70 removed (a server or dead
#: endpoint that must be ANSWERED or REFUSED on the number); listed so that any
#: new site, or a change to one of these, fails the guard.
RELEASED_PORT_SITES = {
    ("test_conversation_provider_acceleration.py", "test_serving_propagates_profile_and_uuid_to_owned_child"):
        "conversation_provider.serve() bind-checks the port itself (a held port is refused by design) "
        "and its child is mocked: nothing ever listens on it",
    ("test_conversation_provider_startup.py", "test_serve_spends_state_verification_time_without_renewing_deadline"):
        "same: serve() bind-checks the port; the child is mocked",
    ("test_conversation_provider_startup.py",
     "test_state_verification_cannot_authorize_child_after_original_startup_deadline"):
        "same: serve() bind-checks the port; the child is mocked",
    ("test_spark_install.py", "launch_fixture"):
        "dead Qwen endpoint that install_support only connects to; a plain helper without teardown "
        "(open follow-up: hold it with held_dead_port)",
}


def test_no_test_module_takes_a_port_by_binding_zero_and_releasing_it():
    found = {(path.relative_to(TESTS).as_posix(), scope)
             for path in sorted(TESTS.rglob("*.py")) for scope in _released_port_sites(path.read_text())}
    assert found == set(RELEASED_PORT_SITES)


@pytest.mark.parametrize("source", [
    # tests/test_hug_backend.py::_free_port before B70
    "def free_port():\n s = socket.socket()\n s.bind(('127.0.0.1', 0))\n port = s.getsockname()[1]\n"
    " s.close()\n return port\n",
    # the tests/test_openclaw_gateway.py fixture before B70
    "def fixture():\n with socket.socket() as server:\n  server.bind(('127.0.0.1', 0))\n"
    "  port = server.getsockname()[1]\n start(port)\n",
    "def unpacked():\n s = socket.socket()\n s.bind(('', 0))\n _, port = s.getsockname()\n s.close()\n"
    " serve(port=port)\n",
], ids=["close-then-return", "with-then-use", "tuple-unpacked"])
def test_the_guard_finds_a_port_released_before_use(source):
    assert len(_released_port_sites(source)) == 1


@pytest.mark.parametrize("source", [
    # a server: it listens on the number, then closes after its use
    "def server():\n s = socket.socket()\n s.bind(('127.0.0.1', 0))\n s.listen()\n"
    " port = s.getsockname()[1]\n use(port)\n s.close()\n check(port)\n",
    # held_dead_port: the number is used while the socket still holds it
    "def held():\n s = socket.socket()\n try:\n  s.bind(('127.0.0.1', 0))\n  yield s.getsockname()[1]\n"
    " finally:\n  s.close()\n",
    "def used_inside():\n with socket.socket() as s:\n  s.bind(('127.0.0.1', 0))\n"
    "  port = s.getsockname()[1]\n  use(port)\n",
    "def fixed():\n s = socket.socket()\n s.bind(('127.0.0.1', 47201))\n port = s.getsockname()[1]\n"
    " s.close()\n use(port)\n",
], ids=["listening-server", "held-dead-port", "used-while-held", "fixed-port"])
def test_the_guard_ignores_a_port_held_while_used(source):
    assert _released_port_sites(source) == set()


def _answers(port: int) -> bool:
    """True when a connection to 127.0.0.1:port completes (refused OR unanswered = False)."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


_HANDED_SERVER = """
import os, sys, time
sys.path.insert(0, sys.argv[3])
from owned_server import take_handed_port
server = take_handed_port(sys.argv[1], int(sys.argv[2]))
server.listen()
print("LISTENING", flush=True)
connection, _ = server.accept()
connection.sendall(str(os.getpid()).encode())
connection.close()
time.sleep(60)
"""


def test_a_handed_over_port_is_never_free_and_closes_with_its_server():
    from owned_server import handed_over_port

    with handed_over_port() as handoff:
        port = handoff.port
        intruder = socket.socket()
        try:
            with pytest.raises(OSError):           # held from the moment it was chosen
                intruder.bind(("127.0.0.1", port))
        finally:
            intruder.close()
        child = subprocess.Popen([sys.executable, "-c", _HANDED_SERVER, handoff.path, str(port), str(TESTS)],
                                 stdout=subprocess.PIPE, text=True)
        try:
            assert child.stdout.readline() == "LISTENING\n"
            assert handoff.handed.wait(10)
            with socket.create_connection(("127.0.0.1", port), timeout=10) as client:
                assert client.recv(64).decode() == str(child.pid)   # the server this test started
        finally:
            child.kill()
            child.wait(timeout=10)
            child.stdout.close()
        assert not _answers(port)                  # no copy outlives the server that took it


def test_a_server_told_another_port_refuses_the_handed_socket():
    from owned_server import OwnedServerError, handed_over_port, take_handed_port

    with handed_over_port() as handoff:
        other = handoff.port - 1 if handoff.port > 1024 else handoff.port + 1
        with pytest.raises(OwnedServerError, match=f"bound to port {handoff.port}, not {other}"):
            take_handed_port(handoff.path, other)


def test_a_claimed_port_is_the_tests_own_and_is_never_handed_out():
    from owned_server import OwnedServerError, handed_over_port, take_handed_port

    with handed_over_port() as handoff:
        foreign = handoff.claim()
        try:
            assert foreign.getsockname() == ("127.0.0.1", handoff.port)
            with pytest.raises(OwnedServerError, match="no socket was handed over"):
                take_handed_port(handoff.path, handoff.port, timeout_s=10)
            assert not handoff.handed.is_set()
            with pytest.raises(OwnedServerError, match="already"):
                handoff.claim()
        finally:
            foreign.close()


def test_a_port_nobody_takes_stays_held_and_unanswered():
    from owned_server import handed_over_port

    with handed_over_port() as handoff:
        assert not _answers(handoff.port)
        intruder = socket.socket()
        try:
            with pytest.raises(OSError):
                intruder.bind(("127.0.0.1", handoff.port))
        finally:
            intruder.close()
        assert not handoff.handed.is_set()
    assert not handoff._thread.is_alive()          # teardown woke and ended the hand-over thread


def test_a_failed_hand_over_keeps_the_port_held(monkeypatch):
    # The server vanished between connecting and receiving: the port must stay
    # held (a dead endpoint), not be dropped and become free.
    import owned_server
    from owned_server import OwnedServerError, handed_over_port, take_handed_port

    def broken_pipe(*args, **kwargs):
        raise BrokenPipeError("server went away")

    with handed_over_port() as handoff:
        monkeypatch.setattr(owned_server.socket, "send_fds", broken_pipe)
        with pytest.raises(OwnedServerError, match="no socket was handed over"):
            take_handed_port(handoff.path, handoff.port, timeout_s=10)
        handoff._thread.join(10)
        assert not handoff._thread.is_alive() and not handoff.handed.is_set()
        intruder = socket.socket()
        try:
            with pytest.raises(OSError):
                intruder.bind(("127.0.0.1", handoff.port))
        finally:
            intruder.close()
