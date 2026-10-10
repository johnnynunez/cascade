"""Test servers on ports the OS assigns, accepted only once they prove who they are.

The GraspGen-X stub and occupancy bridge fixtures used to start their server
on a FIXED port (5599; 5598 / 5599) and called it ready as soon as ANY process
accepted a TCP connection there. Two suites on one host (several worktrees, a
developer shell next to a CI runner) then answered each other's requests:
`zmq.error.Again` in tests/test_graspgenx_backend.py, `KeyError: 'point_cloud'`
in tests/test_occupancy.py (backlog B64).

Contract with the server scripts (`--port 0 --instance-id TOKEN`):

- the server binds a port the OS assigns, which no other live process holds;
- after the bind it prints one stdout line
  ``CASCADE_SERVER_READY {"endpoint", "port", "pid", "instance"}``;
- its health / probe reply carries the same ``pid`` and ``instance``.

`start_owned_server` accepts a server only when the ready line AND one real
protocol round trip both name the process it started and the token it chose.
Readiness IS the ready line: a thread blocks on the child's stdout, so nothing
here sleeps or polls a port.

A "dead" endpoint is `held_dead_port()`: bound, never listening, held for the
whole test. A server that must be TOLD its port up front gets the socket bound
to it handed over by `handed_over_port()` / `take_handed_port()`, so no port is
ever released between choosing it and using it (B70).
"""

from __future__ import annotations

import json
import os
import queue
import secrets
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

READY_PREFIX = "CASCADE_SERVER_READY "


class OwnedServerError(RuntimeError):
    """The server never announced itself, or something other than it answered.

    `proc` is the started process, already killed and reaped."""

    proc: subprocess.Popen | None = None


@dataclass
class OwnedServer:
    proc: subprocess.Popen
    port: int
    instance: str
    stderr_path: Path

    def stop(self) -> bool:
        """Kill and reap the server; True when it was still running, i.e.
        nothing else could have taken its port while the tests used it."""
        alive = self.proc.poll() is None
        if alive:
            self.proc.kill()
        self.proc.wait(timeout=30)
        return alive


def _tail(path: Path, limit: int = 800) -> str:
    try:
        return path.read_text(errors="replace")[-limit:]
    except OSError:
        return ""


def _reap(proc: subprocess.Popen, exc: BaseException) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=30)
    if isinstance(exc, OwnedServerError):
        exc.proc = proc


def _await_ready(proc: subprocess.Popen, lines: queue.Queue, deadline: float,
                 stderr_path: Path) -> dict:
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            raise OwnedServerError(
                f"{proc.args!r} printed no {READY_PREFIX.strip()} line in time; "
                f"stderr: {_tail(stderr_path)}")
        try:
            line = lines.get(timeout=left)
        except queue.Empty:
            continue
        if line is None:
            code = proc.wait(timeout=30)
            raise OwnedServerError(
                f"{proc.args!r} exited ({code}) before its ready line; "
                f"stderr: {_tail(stderr_path)}")
        if line.startswith(READY_PREFIX):
            try:
                return json.loads(line[len(READY_PREFIX):])
            except ValueError as exc:
                raise OwnedServerError(f"unreadable ready line {line!r}") from exc


def spawn_announced(argv: Sequence[str], *, deadline_s: float, stderr_path: Path,
                    instance: str | None = None) -> tuple[subprocess.Popen, dict]:
    """Start ``argv --port 0 [--instance-id TOKEN]`` and return the process and
    its parsed ready line. The process is killed if it never announces."""
    extra = ["--port", "0"] + ([] if instance is None else ["--instance-id", instance])
    with open(stderr_path, "wb") as stderr:
        proc = subprocess.Popen([*argv, *extra], stdout=subprocess.PIPE, stderr=stderr,
                                encoding="utf-8", errors="replace")
    lines: queue.Queue = queue.Queue()
    stdout = proc.stdout
    assert stdout is not None

    def pump() -> None:
        with stdout:
            for line in stdout:
                lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, name=f"owned-server-{proc.pid}", daemon=True).start()
    try:
        return proc, _await_ready(proc, lines, time.monotonic() + deadline_s, stderr_path)
    except BaseException as exc:
        _reap(proc, exc)
        raise


def _require_identity(what: str, info: object, pid: int, instance: str) -> None:
    named = info if isinstance(info, dict) else {}
    if named.get("pid") != pid:
        raise OwnedServerError(f"{what} names pid {named.get('pid')!r}, not the server "
                               f"this test started (pid {pid}): {info!r}")
    if named.get("instance") != instance:
        raise OwnedServerError(f"{what} names instance {named.get('instance')!r}, not "
                               f"this test's token {instance!r}: {info!r}")


def start_owned_server(argv: Sequence[str], *, identify: Callable[[int], object],
                       deadline_s: float, stderr_path: Path) -> OwnedServer:
    """Start a server on a port the OS assigns and accept it only when its
    ready line and ``identify(port)`` -- one round trip over the server's own
    protocol -- name the started process and a fresh random token."""
    instance = secrets.token_hex(8)
    proc, ready = spawn_announced(argv, deadline_s=deadline_s, stderr_path=stderr_path,
                                  instance=instance)
    try:
        _require_identity("ready line", ready, proc.pid, instance)
        port = ready.get("port")
        if type(port) is not int or not 0 < port < 65536:
            raise OwnedServerError(f"ready line names no usable port: {ready!r}")
        try:
            reply = identify(port)
        except Exception as exc:  # noqa: BLE001 -- reported with the server's stderr
            raise OwnedServerError(f"identity round trip to port {port} failed: {exc}; "
                                   f"stderr: {_tail(stderr_path)}") from exc
        _require_identity("protocol reply", reply, proc.pid, instance)
    except BaseException as exc:
        _reap(proc, exc)
        raise
    return OwnedServer(proc, port, instance, stderr_path)


@contextmanager
def held_dead_port() -> Iterator[int]:
    """A loopback port this process holds bound but NOT listening: no
    connection to it ever completes, and no other socket can bind it while the
    test runs -- unlike a literal "nothing listens on 5597" assumption.

    How a connect fails is platform-specific: Linux refuses it at once; on the
    macOS CI runner it is never answered, so the client's own timeout fires.
    A client pointed here must therefore carry a short timeout."""
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        holder.bind(("127.0.0.1", 0))
        yield holder.getsockname()[1]
    finally:
        holder.close()


class PortHandoff:
    """A loopback port held from the moment the OS assigns it until ONE server
    process takes the bound socket over; see `handed_over_port`."""

    def __init__(self, holder: socket.socket, link: socket.socket, path: str):
        self.port: int = holder.getsockname()[1]
        self.path = path                       # give this to the server (env / argv)
        self.handed = threading.Event()        # set once a server holds the socket
        self._holder: socket.socket | None = holder
        self._link = link
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._serve, name=f"port-handoff-{self.port}",
                                        daemon=True)
        self._thread.start()

    def claim(self) -> socket.socket:
        """The held socket, for the test itself (e.g. to listen on the port as
        a foreign server). No server can take it afterwards."""
        with self._lock:
            holder, self._holder = self._holder, None
        if holder is None:
            raise OwnedServerError(f"port {self.port} was already handed over or claimed")
        return holder

    def _serve(self) -> None:
        try:
            connection, _ = self._link.accept()
        except OSError:
            return
        with connection, self._lock:
            holder, self._holder = self._holder, None
            if holder is None:                 # claimed, or closing: hand nothing over
                return
            try:
                socket.send_fds(connection, [b"P"], [holder.fileno()])
            except OSError:                    # the server went away first: keep holding
                self._holder = holder
                return
            self.handed.set()
        holder.close()  # the server's copy is now the only one: the port closes with it

    def close(self) -> None:
        with self._lock:
            holder, self._holder = self._holder, None
        try:  # wake a pending accept(); closing the listener alone does not on Linux
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as wake:
                wake.connect(self.path)
        except OSError:
            pass
        self._thread.join(timeout=10)
        self._link.close()
        if holder is not None:
            holder.close()


@contextmanager
def handed_over_port() -> Iterator[PortHandoff]:
    """A loopback port for a server that must be TOLD its port up front (e.g.
    `openclaw gateway run --port P`), so it cannot bind port 0 itself.

    Binding 0, releasing the socket and letting the server bind the number
    again leaves a window in which another process can take the port. Here it
    is never released: the bound socket waits in this process (no listener, so
    a connect fails exactly as for `held_dead_port`) until the server calls
    `take_handed_port(handoff.path, P)`, which receives that very socket over a
    private Unix socket (SCM_RIGHTS) and checks it is bound to P. This process
    then closes its copy, so the port closes when the server exits."""
    run_dir = tempfile.mkdtemp(prefix="cascade-port-")
    path = os.path.join(run_dir, "handoff")
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    link = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        holder.bind(("127.0.0.1", 0))
        link.bind(path)
        link.listen(1)
        handoff = PortHandoff(holder, link, path)
    except BaseException:
        holder.close()
        link.close()
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    del holder, link  # the hand-over owns both sockets from here on
    try:
        yield handoff
    finally:
        handoff.close()
        shutil.rmtree(run_dir, ignore_errors=True)


def take_handed_port(path: str, port: int, *, timeout_s: float = 10.0) -> socket.socket:
    """Server side of `handed_over_port`: the socket bound to `port`, refused
    unless it really is bound to the port this server was told to use. The
    caller listens on it; stdlib only, so a fake server script can import it."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as link:
        link.settimeout(timeout_s)
        link.connect(path)
        _, fds, _, _ = socket.recv_fds(link, 16, 1)
    if not fds:
        raise OwnedServerError(f"no socket was handed over at {path} (claimed, or taken already)")
    server = socket.socket(fileno=fds[0])
    bound = server.getsockname()[1]
    if bound != port:
        server.close()
        raise OwnedServerError(f"the handed socket is bound to port {bound}, not {port}, "
                               "the port this server was told to use")
    return server
