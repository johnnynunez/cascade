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
"""

from __future__ import annotations

import json
import queue
import secrets
import socket
import subprocess
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
    """A loopback port this process holds bound but NOT listening: every
    connection is refused, and no other socket can bind it while the test
    runs -- unlike a literal "nothing listens on 5597" assumption."""
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        holder.bind(("127.0.0.1", 0))
        yield holder.getsockname()[1]
    finally:
        holder.close()
