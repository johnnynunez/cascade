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

    async def connect(self, session):
        import aiohttp
        headers = {}
        if self.config.token_env:
            token = os.environ.get(self.config.token_env)
            if not token:
                raise ValueError("configured provider credential is unavailable")
            headers["Authorization"] = "Bearer " + token
        self._client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        try:
            self._ws = await asyncio.wait_for(self._client.ws_connect(
                self.config.url, headers=headers, heartbeat=20,
                max_msg_size=self.config.max_event_bytes), self.config.connect_timeout_s)
            event = await asyncio.wait_for(self.receive(), self.config.connect_timeout_s)
            if event.get("type") != "session.created":
                raise ValueError("provider did not create a Realtime session")
            await self.send({"type": "session.update", "session": session})
            event = await asyncio.wait_for(self.receive(), self.config.connect_timeout_s)
            if event.get("type") != "session.updated":
                raise ValueError("provider did not acknowledge session configuration")
            actual = event.get("session", {})
            for direction in ("input", "output"):
                fmt = actual.get("audio", {}).get(direction, {}).get("format", {})
                if fmt.get("type") != "audio/pcm" or fmt.get("rate") != 24000:
                    raise ValueError("provider did not admit requested PCM format")
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
        stages = []
        if self._ws is not None:
            async def close_socket():
                await asyncio.wait_for(self._ws.close(), 1)
            stages.append(await close_stage("websocket", close_socket))
        if self._client is not None:
            stages.append(await close_stage("client", self._client.close))
            if stages[-1]["ok"]:
                # The client owns the WebSocket transport too.
                self._ws = self._client = None
        elif stages and stages[-1]["ok"]:
            self._ws = None
        self._closure_receipt = retain_teardown_attempt(self._closure_receipt, teardown_receipt(stages))
        return self._closure_receipt
