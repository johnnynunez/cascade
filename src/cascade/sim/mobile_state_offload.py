"""Owner side of the opt-in off-GIL ``state()`` reader (shared MicroDuck owner, B12a).

Client ``state()`` polls (about one per robot per step) used to be answered by the owner's TCP
endpoint threads, on the owner's GIL, inflating whichever phase the simulation thread was in.
With ``--state-reader process`` the owner keeps every channel's TCP endpoint, ``hello``, commands,
renewals, stop, reset and frames exactly as before; only the polling of a reader channel moves:

1. ``StatePublishingController`` mirrors its own ``state()`` reply (``marshal``) into the robot's
   shared-memory slot under the controller lock after every completed step (``publish``) and
   every permission change (stop, reset, admission, fault, epoch, script, completion), so a stop
   ACK is never followed by a pre-stop reply. Polling paths (``control_at``, ``watchdog``,
   ``renew``) republish only when the reply changed.
2. ``StateServerProcess`` creates the slots, spawns ``mobile_state_server.py`` (standard library
   only, ``sys.executable -I``) and receives what it hands back.
3. After a reader channel's first ``state`` reply, ``MobileBridgeServer`` hands the connection
   descriptor to that process (SCM_RIGHTS); a ``frame`` there comes back with its pending request
   and stays on the owner. The owner's connection slot stays held while the reader is away.

Replies are byte-identical to the in-GIL path at the same clock: the server re-encodes the
owner's own dict and recomputes only the two age fields from the slot's publish time, never
refreshing it. A failed slot write, a dead server or a closed slot fails closed (explicit
refusal, closed connections); none of this changes a limit, a deadline or a verdict.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import functools
import marshal
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from .mobile_bridge import MobileBridgeController, MobileBridgeServer
from .mobile_state_server import IMPLEMENTATION, SlotSegment, recv_message, send_message

SERVER_PATH = Path(__file__).resolve().with_name("mobile_state_server.py")
READ_ONLY = ("hello", "state")
PUBLISHING = ("script", "publish", "command_velocity", "scale_velocity", "renew", "watchdog", "owner_disconnected",
              "control_at", "stop", "reset_stop", "fault", "begin_epoch")
_ON_CHANGE = frozenset({"renew", "watchdog", "control_at"})  # polled; republish only if the reply changed
PAYLOAD_BYTES = 1 << 20


class StatePublishingController(MobileBridgeController):
    """A ``MobileBridgeController`` whose ``state()`` reply is mirrored into a shared-memory slot.

    Detached (no slot) it is the base class, call for call. Attached, every method in
    ``PUBLISHING`` republishes under the controller lock before returning, so the slot never
    lags an ACK; ``_mirror_state`` never raises into the control path (a failure marks the slot
    unavailable and is recorded in ``slot_errors``).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slot = self._slot_index = None
        self._slot_depth = 0
        self._slot_dirty = False
        self.slot_publications = 0
        self.slot_errors = []

    def attach_state_slot(self, segment, index):
        with self._lock:
            if self._slot is not None:
                raise RuntimeError("state slot already attached")
            hello = self.hello()
            if tuple(segment.identities[index]) != (hello["robot_id"], hello["model_identity_sha256"]):
                raise ValueError("state slot identity differs from this controller")
            self._slot, self._slot_index = segment, index
            self._mirror_state()

    def _slot_signature(self):
        active = self._active
        return (self._epoch, self._generation, self._latched, self._fault,
                None if active is None else active["command_id"], self._last_completed,
                self._script is None, self._state_wall, None if self._state is None else self._state.get("step"))

    def _mirror_state(self):
        slot = self._slot
        if slot is None:
            return
        try:
            reply = self.state()
            written = slot.write(self._slot_index, marshal.dumps(reply), state_wall=self._state_wall,
                                 step=None if self._state is None else self._state["step"],
                                 generation=self._generation)
            if written:
                self.slot_publications += 1
            elif not slot._closed:
                self._slot_error("state reply exceeds the off-GIL slot capacity")
        except Exception as exc:  # never into the control path: refuse the slot instead
            self._slot_error(f"{type(exc).__name__}: {exc}")
            try:
                slot.mark_unavailable(self._slot_index, f"state publication failed: {type(exc).__name__}")
            except Exception:
                pass

    def _slot_error(self, reason):
        if len(self.slot_errors) < 16:
            self.slot_errors.append(reason[:240])


def _publishing(name):
    method = getattr(MobileBridgeController, name)
    on_change = name in _ON_CHANGE

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self._slot is None:
            return method(self, *args, **kwargs)
        with self._lock:
            outer = self._slot_depth == 0
            before = self._slot_signature() if outer and on_change else None
            self._slot_depth += 1
            try:
                return method(self, *args, **kwargs)
            finally:
                self._slot_depth -= 1
                if not on_change:
                    self._slot_dirty = True
                if outer and (self._slot_dirty or self._slot_signature() != before):
                    self._slot_dirty = False
                    self._mirror_state()
    return wrapper


for _name in PUBLISHING:
    setattr(StatePublishingController, _name, _publishing(_name))
del _name


@dataclass(frozen=True)
class PendingRequest:
    """One complete request a handed-back reader still owes a reply to."""
    line: bytes
    deadline: float
    received_wall: float


class _ReaderOffload:
    def __init__(self, process, robot):
        self.process, self.robot = process, robot

    def handoff(self, conn):
        return self.process._handoff(self.robot, conn)


class StateServerProcess:
    """Slots + the reader-server process + what it hands back, for one owner's endpoints."""

    START_TIMEOUT_S = 10.
    STOP_TIMEOUT_S = 5.

    def __init__(self, controllers, *, payload_bytes=PAYLOAD_BYTES, executable=None):
        controllers = dict(controllers)
        if not hasattr(socket, "send_fds"):
            raise RuntimeError("the off-GIL state reader needs Unix descriptor passing")
        for robot_id, controller in controllers.items():
            if not isinstance(controller, StatePublishingController):
                raise TypeError("the off-GIL state reader needs StatePublishingController controllers")
            if controller._clock is not time.monotonic:
                raise ValueError("off-GIL replies are aged on time.monotonic; this controller uses another clock")
            if controller.hello()["robot_id"] != robot_id:
                raise ValueError("controller identity differs from its endpoint")
        self.robots = tuple(controllers)
        self.controllers = controllers
        self.counters = dict(handoffs=0, handoff_refused=0, returned=0, released=0)
        self.errors = []
        self.server_stats = self.exit_code = self.pid = None
        self.process = self.control = self._receiver = None
        self._servers = {}
        self._away = {robot: 0 for robot in self.robots}
        self._lock = threading.Lock()  # counters, the away counts and the owner->server direction
        self._closed = self._dead = False
        self.slots = SlotSegment.create([(robot, controllers[robot].hello()["model_identity_sha256"])
                                         for robot in self.robots], payload_bytes=payload_bytes)
        try:
            for index, robot in enumerate(self.robots):
                controllers[robot].attach_state_slot(self.slots, index)
            self._spawn(executable or sys.executable)
        except BaseException as exc:
            try:
                self.close()
            except BaseException as cleanup:
                exc.add_note(f"state reader cleanup failed: {type(cleanup).__name__}: {cleanup}")
            raise

    def _spawn(self, executable):
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self.control = parent
        try:
            # -I: no PYTHONPATH, user site or script directory; the server is standard library only.
            # Its stdout is never the owner's (harnesses parse the owner's stdout).
            self.process = subprocess.Popen([executable, "-I", str(SERVER_PATH), "--control-fd", str(child.fileno())],
                                            pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL,
                                            stdout=subprocess.DEVNULL, close_fds=True)
        finally:
            child.close()
        protocol = dict(max_request_bytes=MobileBridgeServer.MAX_REQUEST_BYTES,
                        max_response_bytes=MobileBridgeServer.MAX_RESPONSE_BYTES,
                        io_timeout_s=MobileBridgeServer.IO_TIMEOUT_S,
                        reader_ops=sorted(MobileBridgeServer.READER_OPS), forbidden=MobileBridgeServer.FORBIDDEN)
        send_message(parent, {"op": "config", "segment": self.slots.name,
                              "robots": [list(item) for item in self.slots.identities],
                              "python": list(sys.version_info[:2]), "marshal_version": marshal.version,
                              "protocol": protocol})
        parent.settimeout(self.START_TIMEOUT_S)
        try:
            message, fd = recv_message(parent)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"state server did not start: {type(exc).__name__}: {exc}") from exc
        if fd is not None:
            os.close(fd)
        if message is None or message.get("op") != "ready" or message.get("pid") != self.process.pid:
            raise RuntimeError(f"state server refused to start: {message}")
        parent.settimeout(None)
        self.pid = self.process.pid
        self._receiver = threading.Thread(target=self._receive, name="state-server-receiver", daemon=True)
        self._receiver.start()

    def offload(self, robot):
        if robot not in self._away:
            raise ValueError("unknown robot identity")
        return _ReaderOffload(self, robot)

    def bind(self, robot, server):
        if robot not in self._away or robot in self._servers:
            raise ValueError("each robot endpoint binds once")
        self._servers[robot] = server

    def _handoff(self, robot, conn):
        """Owner RPC worker: move a reader to the server; False keeps it on the owner (counted)."""
        with self._lock:
            if self._closed or self._dead or robot not in self._servers or self.process.poll() is not None:
                self.counters["handoff_refused"] += 1
                return False
            try:
                send_message(self.control, {"op": "adopt", "robot": self.robots.index(robot)}, conn.fileno())
            except (OSError, ValueError) as exc:
                self.counters["handoff_refused"] += 1
                self._error(f"handoff: {type(exc).__name__}: {exc}")
                return False
            self._away[robot] += 1
            self.counters["handoffs"] += 1
            return True

    def _receive(self):
        while True:
            try:
                message, fd = recv_message(self.control)
            except (OSError, ValueError) as exc:
                if not self._closed:
                    self._error(f"control: {type(exc).__name__}: {exc}")
                break
            if message is None:
                break
            op = message.get("op")
            try:
                if op == "return":
                    owned, fd = fd, None
                    self._returned(message, owned)
                elif op == "released":
                    self._released(message)
                elif op == "stats":
                    self.server_stats = message
            except (KeyError, TypeError, ValueError) as exc:
                self._error(f"control message {op!r}: {type(exc).__name__}: {exc}")
            finally:
                if fd is not None:
                    os.close(fd)
        # The server is gone: every reader it still held went with it. Free their owner slots.
        with self._lock:
            self._dead = True
            away, self._away = self._away, {robot: 0 for robot in self.robots}
        for robot, count in away.items():
            for _ in range(count):
                self._servers[robot].release_offloaded()

    def _robot(self, message):
        index = message.get("robot")
        if type(index) is not int or not 0 <= index < len(self.robots):
            raise ValueError("unknown robot slot")
        return self.robots[index]

    def _released(self, message):
        robot = self._robot(message)
        with self._lock:
            if self._away[robot] <= 0:
                return
            self._away[robot] -= 1
            self.counters["released"] += 1
        self._servers[robot].release_offloaded()

    def _returned(self, message, fd):
        """A reader the server handed back (``frame``): the owner serves it from now on."""
        conn = None if fd is None else socket.socket(fileno=fd)  # owns the descriptor from here
        try:
            robot = self._robot(message)
        except ValueError:
            if conn is not None:
                conn.close()
            raise
        with self._lock:
            if self._away[robot] <= 0:
                if conn is not None:
                    conn.close()
                return
            self._away[robot] -= 1
            self.counters["returned"] += 1
        server = self._servers[robot]
        if conn is None:
            server.release_offloaded()
            return
        try:
            pending = PendingRequest(base64.b64decode(message["line"], validate=True),
                                     float(message["deadline"]), float(message["received_wall"]))
        except (KeyError, TypeError, ValueError):
            conn.close()
            server.release_offloaded()
            raise
        server.adopt(conn, pending)

    def _error(self, reason):
        if len(self.errors) < 16:
            self.errors.append(reason[:240])

    def receipt(self):
        controllers = self.controllers
        return {"mode": "process", "implementation": IMPLEMENTATION, "server_pid": self.pid,
                "server_exit_code": self.exit_code, "payload_capacity_bytes": self.slots.capacity,
                "slots": len(self.robots), **self.counters,
                "publications": {robot: controllers[robot].slot_publications for robot in self.robots},
                "publish_errors": [error for robot in self.robots for error in controllers[robot].slot_errors],
                "server": None if self.server_stats is None else self.server_stats.get("stats"),
                "errors": list(self.errors)}

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        first = None
        steps = []
        if self.slots is not None:
            steps.append(lambda: self.slots.mark_closed("owner closed the off-GIL state reader"))
        if self.control is not None and self.process is not None:
            steps.append(lambda: send_message(self.control, {"op": "close"}))
        if self.process is not None:
            def stop():
                try:
                    self.process.wait(self.STOP_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(self.STOP_TIMEOUT_S)
                    raise
                finally:
                    self.exit_code = self.process.returncode
            steps.append(stop)
        if self._receiver is not None:
            steps.append(lambda: self._receiver.join(self.STOP_TIMEOUT_S))
        if self.control is not None:
            steps.append(self.control.close)
        if self.slots is not None:
            steps.append(lambda: self.slots.close(unlink=True))
        for step in steps:
            try:
                step()
            except BaseException as exc:
                if first is None:
                    first = exc
        if first is not None:
            raise first
