"""Client of a humanoid loco-manipulation bridge (JSON lines over loopback TCP).

The bridge owns the simulator and the learned policy client (``scripts/arena_locomanip_bridge.py``
runs Isaac Lab-Arena with a GR00T policy server in its own environment); CASCADE never imports
either. One client session at a time per bridge; a session starts from a freshly reset scene.

Messages (one JSON object per line):

==========================  =====================================================================
client -> bridge            bridge -> client
==========================  =====================================================================
(connect)                   ``hello``: protocol, env, physics, policy, objects, destinations,
                            joint model, control rate; or ``busy`` (another session is active)
``session`` {seed}          ``ready`` {state}: scene reset, measured state
``observe``                 ``state`` {state}: measured state, no physics step
``segment`` {id, object,    one ``step`` {i, state, action} per control step, each answered by
destination, instruction,   ``gate`` {i, verdict: pass|stop, clamp: [[joint index, target], ...]};
max_steps, rest_steps}      the bridge applies the action only after ``pass``, with every listed
                            joint target replaced by the clamped value (finger targets moved
                            onto their joint range). Then ``segment_end`` {reason, steps, final,
                            arena, recording}
``close``                   (closes)
==========================  =====================================================================

A protocol violation, a silent bridge or a dropped connection raises :class:`BridgeError`;
callers treat it as "nothing verified".
"""
from __future__ import annotations

import json
import socket
import time

PROTOCOL = 2  # 2: the gate reply carries finger-target clamps the bridge must apply


class BridgeError(RuntimeError):
    """The bridge is unavailable, silent, busy or broke the protocol."""


class BridgeBusy(BridgeError):
    pass


class BridgeClient:
    def __init__(self, host: str, port: int, *, connect_timeout_s: float = 5.0, reply_timeout_s: float = 30.0):
        self.host, self.port = host, int(port)
        self.reply_timeout_s = float(reply_timeout_s)
        try:
            self._sock = socket.create_connection((host, self.port), timeout=connect_timeout_s)
        except OSError as exc:
            raise BridgeError(f"bridge {host}:{port} unreachable: {exc}") from exc
        self._sock.settimeout(self.reply_timeout_s)
        self._rfile = self._sock.makefile("r", encoding="utf-8", newline="\n")
        self.hello = self.receive(timeout_s=connect_timeout_s)
        if self.hello.get("type") == "busy":
            self.close()
            raise BridgeBusy(f"bridge {host}:{port} is serving another session")
        if self.hello.get("type") != "hello" or self.hello.get("protocol") != PROTOCOL:
            self.close()
            raise BridgeError(f"bridge {host}:{port} sent no protocol-{PROTOCOL} hello")

    def send(self, message: dict) -> None:
        try:
            self._sock.sendall((json.dumps(message, allow_nan=False) + "\n").encode())
        except (OSError, ValueError) as exc:
            raise BridgeError(f"bridge send failed: {exc}") from exc

    def receive(self, *, timeout_s: float | None = None) -> dict:
        self._sock.settimeout(self.reply_timeout_s if timeout_s is None else timeout_s)
        try:
            line = self._rfile.readline()
        except (OSError, ValueError) as exc:
            raise BridgeError(f"bridge silent or dropped: {exc}") from exc
        if not line:
            raise BridgeError("bridge closed the connection")
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BridgeError(f"bridge sent malformed JSON: {exc}") from exc
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            raise BridgeError("bridge message is not a typed object")
        if message["type"] == "error":
            raise BridgeError(f"bridge error: {message.get('error')}")
        return message

    def request(self, message: dict, expect: str, *, timeout_s: float | None = None) -> dict:
        self.send(message)
        reply = self.receive(timeout_s=timeout_s)
        if reply["type"] != expect:
            raise BridgeError(f"expected {expect!r} from the bridge, got {reply['type']!r}")
        return reply

    def close(self) -> None:
        try:
            self.send({"type": "close"})
        except BridgeError:
            pass
        for item in (self._rfile, self._sock):
            try:
                item.close()
            except OSError:
                pass


def connect_first_free(host: str, ports, *, wait_s: float = 0.0, **kwargs) -> BridgeClient:
    """Connect to the first bridge in ``ports`` that is not serving another session."""
    deadline = time.monotonic() + wait_s
    while True:
        errors = []
        for port in ports:
            try:
                return BridgeClient(host, port, **kwargs)
            except BridgeError as exc:
                errors.append(str(exc))
        if time.monotonic() >= deadline:
            raise BridgeError("; ".join(errors) or "no bridge port configured")
        time.sleep(2.0)
