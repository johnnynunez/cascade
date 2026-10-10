#!/usr/bin/env python
"""An openpi / LingBot-VLA compatible policy server that runs anywhere -- no
GPU, no weights, no model code.

    python scripts/serve_vla_stub.py [--host 127.0.0.1] [--port 8000]
                                     [--chunks chunks.json] [--action-key actions]

WHY THIS EXISTS. `grasp.executor: vla` (B49, opt-in) serves grasp_object from
a VLA policy server: openpi's `serve_policy.py` or LingBot-VLA-v2's
`python -m deploy.lingbot_vla_v2_policy`. Both need a CUDA box, their own
environment and multi-GB checkpoints, so the client, the chunk conversion, the
harness gating, the deadlines and the stop latch would otherwise only ever run
on the GPU host. This speaks the SAME wire and replays SCRIPTED action chunks,
so the whole route runs and is tested on any machine. It is a TEST DOUBLE,
not a policy: it does not look at the image and learns nothing.

Protocol (openpi `WebsocketPolicyServer` == LingBot `deploy/websocket_policy_server.py`):

  on connect   server sends one msgpack frame: the metadata dict
  request      msgpack obs dict (openpi msgpack-numpy arrays)
  reply        msgpack {<action key>: (H, D) float32, "server_timing": {"infer_ms", "prev_total_ms"}}
  error        a TEXT frame with the traceback, then close 1011
  GET /healthz 200 "OK"
  LingBot reset   an obs with `reset: true` gets {"action": None}

`--chunks` is a JSON list of chunks (each a list of rows); request N gets chunk
N and the last chunk repeats. Without it the stub HOLDS: every reply is one
row equal to the observed state (joints + jaw opening), so a wiring smoke test
moves nothing and closes nothing.
"""

from __future__ import annotations

import argparse
import http
import json
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

try:
    from cascade.grasping.vla_client import pack, unpack
except ImportError:  # run from a source checkout without an install
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cascade.grasping.vla_client import pack, unpack


class ScriptedPolicy:
    """Replays `chunks` (request N -> chunk N, the last one repeats).

    Test hooks: `requests` (every observation received), `extra` (keys merged
    into each reply -- e.g. a self-reported `success`), `delays` ({request
    number: seconds to sleep before replying}), `on_call(obs, n)` (runs as
    request n arrives), `respond(obs)` (replaces the script), `track(arm)`
    (records `len(arm.commands)` in `command_marks` as each request arrives).
    """

    def __init__(self, chunks=(), *, action_key: str = "actions", state_key: str = "observation/state"):
        self.chunks = list(chunks)
        self.action_key = action_key
        self.state_key = state_key
        self.requests: list[dict] = []
        self.resets: list[dict] = []
        self.extra: dict = {}
        self.delays: dict[int, float] = {}
        self.on_call = None
        self.respond = None
        self.command_marks: list[int] = []
        self._arm = None
        self._lock = threading.Lock()

    def track(self, arm) -> None:
        self._arm = arm

    def __call__(self, obs: dict) -> dict:
        if obs.get("reset"):
            self.resets.append(obs)
            return {"action": None}
        with self._lock:
            self.requests.append(obs)
            call = len(self.requests)
            if self._arm is not None:
                self.command_marks.append(len(self._arm.commands))
        if self.on_call is not None:
            self.on_call(obs, call)
        delay = self.delays.get(call)
        if delay:
            time.sleep(float(delay))
        if self.respond is not None:
            return self.respond(obs)
        if not self.chunks:
            state = np.asarray(obs[self.state_key], dtype=np.float32).reshape(1, -1)
            return {self.action_key: state, **self.extra}
        chunk = self.chunks[min(call, len(self.chunks)) - 1]
        return {self.action_key: np.asarray(chunk, dtype=np.float32), **self.extra}


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


class StubPolicyServer:
    """websockets' sync server on a daemon thread; `port=0` = ephemeral."""

    def __init__(self, policy, host: str = "127.0.0.1", port: int = 0, metadata: dict | None = None):
        self.policy = policy
        self.host, self.port = host, int(port)
        self.metadata = dict(metadata or {"policy": "scripted-stub"})
        self._server = None
        self._thread = None

    def start(self) -> "StubPolicyServer":
        from websockets.sync.server import serve

        self._server = serve(self._handler, self.host, self.port, compression=None,
                             max_size=None, process_request=_health_check)
        self.port = int(self._server.socket.getsockname()[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True,
                                        name=f"vla-stub:{self.port}")
        self._thread.start()
        return self

    def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _handler(self, ws) -> None:
        from websockets.exceptions import ConnectionClosed

        try:
            ws.send(pack(self.metadata))
        except ConnectionClosed:
            return
        prev_total = None
        while True:
            try:
                raw = ws.recv()
            except ConnectionClosed:
                return
            start = time.monotonic()
            try:
                obs = unpack(raw)
                t_infer = time.monotonic()
                action = dict(self.policy(obs))
                action["server_timing"] = {"infer_ms": (time.monotonic() - t_infer) * 1e3}
                if prev_total is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total * 1e3
                ws.send(pack(action))
                prev_total = time.monotonic() - start
            except ConnectionClosed:
                return
            except Exception:  # noqa: BLE001 -- the protocol's error path, like openpi
                try:
                    ws.send(traceback.format_exc())
                    ws.close(code=1011, reason="Internal server error. Traceback included in previous frame.")
                except Exception:  # noqa: BLE001 -- the client may already be gone
                    pass
                return


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--chunks", type=Path, help="JSON list of chunks (lists of rows); default: hold")
    ap.add_argument("--action-key", default="actions")
    ap.add_argument("--state-key", default="observation/state")
    args = ap.parse_args(argv)
    chunks = json.loads(args.chunks.read_text()) if args.chunks else []
    policy = ScriptedPolicy(chunks, action_key=args.action_key, state_key=args.state_key)
    server = StubPolicyServer(policy, args.host, args.port,
                              metadata={"policy": "scripted-stub", "chunks": len(chunks)}).start()
    print(f"[vla-stub] ws://{args.host}:{server.port} "
          f"({len(chunks) or 'hold'} scripted chunk(s), action key {args.action_key!r})", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
