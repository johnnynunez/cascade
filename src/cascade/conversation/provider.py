"""HF speech-to-speech GA Realtime WebSocket transport (optional aiohttp)."""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

from .lifecycle import close_stage
from ..lifecycle import retain_teardown_attempt, teardown_receipt

HF_PROTOCOL_REVISION = "411399d34555b2169823a6eaeb7f8ff192db89db"


class RealtimeProvider(Protocol):
    async def connect(self, session: dict) -> None: ...
    async def send(self, event: dict) -> None: ...
    async def receive(self) -> dict: ...
    async def close(self) -> dict | None: ...


@dataclass(frozen=True)
class RealtimeConfig:
    url: str
    token_env: str | None = None
    connect_timeout_s: float = 10
    io_timeout_s: float = 5
    max_event_bytes: int = 262144
    release_contract: str | None = None

    def __post_init__(self):
        url = urlsplit(self.url)
        if url.scheme not in {"ws", "wss"} or not url.hostname or url.username or url.password or url.fragment:
            raise ValueError("expected a WebSocket URL without embedded credentials or fragment")
        if url.scheme == "ws" and url.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("unencrypted provider WebSocket must be loopback")
        if not 0 < self.connect_timeout_s <= 60 or not 0 < self.io_timeout_s <= 30:
            raise ValueError("invalid network deadline")
        if type(self.max_event_bytes) is not int or not 4096 <= self.max_event_bytes <= 1048576:
            raise ValueError("invalid event bound")
        if self.token_env is not None and (not self.token_env.isidentifier() or len(self.token_env) > 128):
            raise ValueError("invalid credential environment variable")
        if self.release_contract is not None and (type(self.release_contract) is not str
                                                  or self.release_contract != "hf_pool"):
            raise ValueError("unknown provider release contract")
        if self.release_contract == "hf_pool":
            from .provider_pool import pool_url
            pool_url(self.url)


class RealtimeWebSocket:
    """No allocation service, model download, implicit reconnect or robot identity.

    Authentication is read from the configured server environment only. Endpoint
    query strings and remote exception text are never returned to a browser.
    """
    def __init__(self, config: RealtimeConfig):
        self.config = config
        self._client = self._ws = None
        self._send_lock = asyncio.Lock()
        self._closure_receipt = None
        self._pool_binding = None
        self._pool_claimed = False
        self._pool_released = False
        self._pool_close_deadline = None
        self._headers = {}
        self._close_lock = asyncio.Lock()

    async def connect(self, session):
        import aiohttp
        headers = {}
        if self.config.token_env:
            token = os.environ.get(self.config.token_env)
            if not token:
                raise ValueError("configured provider credential is unavailable")
            headers["Authorization"] = "Bearer " + token
        client_options = {}
        if self.config.release_contract == "hf_pool":
            trace = aiohttp.TraceConfig()
            async def reject_redirect(*_):
                raise ValueError("HF pool WebSocket handshake must not redirect")
            trace.on_request_redirect.append(reject_redirect)
            client_options["trace_configs"] = [trace]
        self._client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None), **client_options)
        self._headers = headers
        connect_deadline = asyncio.get_running_loop().time() + self.config.connect_timeout_s
        def connect_remaining():
            if self.config.release_contract is None:
                return self.config.connect_timeout_s  # Existing generic transport contract.
            from .provider_pool import remaining
            return remaining(connect_deadline)
        try:
            timeout = connect_remaining()
            if self.config.release_contract == "hf_pool":
                # Even an interrupted upgrade can have reached the peer. Until
                # session.created is bound, release remains unproved.
                self._pool_claimed = True
            self._ws = await asyncio.wait_for(self._client.ws_connect(
                self.config.url, headers=headers, heartbeat=20,
                max_msg_size=self.config.max_event_bytes), timeout)
            timeout = connect_remaining()
            event = await asyncio.wait_for(self.receive(), timeout)
            if event.get("type") != "session.created":
                raise ValueError("provider did not create a Realtime session")
            if self.config.release_contract == "hf_pool":
                from .provider_pool import pool_url, read_pool
                identity = event.get("session", {}).get("id")
                if type(identity) is not str or not 0 < len(identity) <= 256:
                    raise ValueError("provider did not expose a bound pool session identity")
                snapshot = await read_pool(self._client, pool_url(self.config.url), headers,
                                           connect_deadline, self.config.max_event_bytes)
                matches = [u for u in snapshot["units"] if u["session_id"] == identity]
                if len(matches) != 1 or matches[0]["state"] != "active":
                    raise ValueError("provider session does not identify an active pool unit")
                self._pool_binding = {"size": snapshot["size"], "index": matches[0]["index"],
                                      "indices": sorted(u["index"] for u in snapshot["units"]),
                                      "session_id": identity}
            update = {"type": "session.update", "session": session}
            if self.config.release_contract is None:
                await self.send(update)
            else:
                timeout = connect_remaining()
                await asyncio.wait_for(self.send(update), timeout)
            timeout = connect_remaining()
            event = await asyncio.wait_for(self.receive(), timeout)
            if event.get("type") != "session.updated":
                raise ValueError("provider did not acknowledge session configuration")
            actual = event.get("session", {})
            for direction in ("input", "output"):
                fmt = actual.get("audio", {}).get(direction, {}).get("format", {})
                if fmt.get("type") != "audio/pcm" or fmt.get("rate") != 24000:
                    raise ValueError("provider did not admit requested PCM format")
            connect_remaining()
        except BaseException:
            await self.close()
            raise

    async def send(self, event):
        payload = json.dumps(event, allow_nan=False, separators=(",", ":"))
        if len(payload.encode()) > self.config.max_event_bytes:
            raise ValueError("outgoing provider event exceeds limit")
        async with self._send_lock:
            if self._ws is None or self._ws.closed:
                raise ConnectionError("provider disconnected")
            await asyncio.wait_for(self._ws.send_str(payload), self.config.io_timeout_s)

    async def receive(self):
        import aiohttp
        if self._ws is None:
            raise ConnectionError("provider disconnected")
        message = await self._ws.receive()
        if message.type != aiohttp.WSMsgType.TEXT:
            raise ConnectionError("provider disconnected or sent a non-JSON event")
        event = json.loads(message.data, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise TypeError("invalid provider event")
        if event["type"] == "error":
            raise ValueError("provider reported a protocol error")
        return event

    async def close(self):
        async with self._close_lock:
            return await self._close()

    async def _close(self):
        stages = []
        if self._pool_close_deadline is None:
            self._pool_close_deadline = asyncio.get_running_loop().time() + self.config.io_timeout_s
        deadline = self._pool_close_deadline
        if self._ws is not None:
            async def close_socket():
                timeout = 1
                if self.config.release_contract is not None:
                    from .provider_pool import remaining
                    timeout = min(timeout, remaining(deadline))
                await asyncio.wait_for(self._ws.close(), timeout)
            stages.append(await close_stage("websocket", close_socket))
        if self._pool_claimed and not self._pool_released:
            async def release_pool():
                from .provider_pool import pool_url, wait_released
                if self._pool_binding is None or self._client is None:
                    raise ValueError("provider pool session was not bound before close")
                result = await wait_released(self._client, pool_url(self.config.url), self._headers,
                                             self._pool_binding, deadline, self.config.max_event_bytes)
                self._pool_released = True
                return result
            stages.append(await close_stage("pool_release", release_pool))
        if self._client is not None:
            stages.append(await close_stage("client", self._client.close))
            if stages[-1]["ok"]:
                # The client owns the WebSocket transport too.
                self._ws = self._client = None
        elif stages and stages[-1]["ok"]:
            self._ws = None
        self._closure_receipt = retain_teardown_attempt(self._closure_receipt, teardown_receipt(stages))
        return self._closure_receipt
