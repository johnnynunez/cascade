"""Authenticated, loopback-only HTTP/media gateway; optional aiohttp imports."""
from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import secrets
from importlib.resources import files

from .media import QueueMediaIO, decode_pcm
from .session import ConversationSession


class ConversationGateway:
    def __init__(self, domain, provider_factory, *, token=None):
        self.domain, self.provider_factory = domain, provider_factory
        self.token = token or secrets.token_urlsafe(32)
        if len(self.token) < 32:
            raise ValueError("gateway token must contain at least 32 characters")
        self.session = None
        self._socket = None
        self._creating = False
        self._ticket = None
        self._ticket_deadline = 0
        self._ticket_task = None
        self._origin = None
        self._runner = None

    def _origin_valid(self, request):
        origin = request.headers.get("Origin")
        # Non-browser clients still need the unguessable Authorization token.
        return origin is None or origin == self._origin

    def _authorized(self, request):
        return self._origin_valid(request) and hmac.compare_digest(
            request.headers.get("Authorization", ""), "Bearer " + self.token)

    def application(self):
        from aiohttp import web

        @web.middleware
        async def protect(request, handler):
            if request.path.startswith("/api/") and request.path != "/api/media" and not self._authorized(request):
                raise web.HTTPForbidden(text="authorization required")
            try:
                response = await handler(request)
            except (ValueError, json.JSONDecodeError, asyncio.QueueFull):
                raise web.HTTPBadRequest(text="invalid request") from None
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Content-Security-Policy"] = "default-src 'self'; connect-src 'self'; media-src 'self' blob:; object-src 'none'; frame-ancestors 'none'"
            return response

        app = web.Application(middlewares=[protect], client_max_size=262144)
        app.router.add_get("/", self._static)
        app.router.add_get("/app.js", self._static)
        app.router.add_get("/playback.js", self._static)
        app.router.add_get("/capture.js", self._static)
        app.router.add_get("/app.css", self._static)
        app.router.add_post("/api/session", self._create)
        app.router.add_delete("/api/session", self._disconnect)
        app.router.add_post("/api/stop", self._stop)
        app.router.add_post("/api/reset", self._reset)
        app.router.add_get("/api/status", self._status)
        app.router.add_get("/api/media", self._media)
        return app

    async def start(self, *, host="127.0.0.1", port=8780):
        from aiohttp import web
        if not ipaddress.ip_address(host).is_loopback or type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("gateway binds an explicit loopback IP only")
        self._runner = web.AppRunner(self.application(), access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host, port)
        await site.start()
        address = self._runner.addresses[0]
        rendered_host = f"[{host}]" if ":" in host else host
        self._origin = f"http://{rendered_host}:{address[1]}"
        return self._origin

    async def _static(self, request):
        from aiohttp import web
        name = "index.html" if request.path == "/" else request.path[1:]
        data = files("cascade.conversation").joinpath("static", name).read_bytes()
        content_type = "text/html" if name.endswith("html") else "text/css" if name.endswith("css") else "application/javascript"
        return web.Response(body=data, content_type=content_type)

    async def _create(self, request):
        from aiohttp import web
        if self._creating or (self.session is not None and not self.session.closed):
            raise web.HTTPConflict(text="session already exists")
        # Reserve before the first await: two slow HTTP bodies must not both
        # pass admission and overwrite the sole owned provider session.
        self._creating = True
        try:
            body = await request.json()
            if body != {"robot_id": self.domain.robot_id}:
                raise web.HTTPBadRequest(text="robot identity mismatch")
            media = QueueMediaIO()
            session = ConversationSession(self.domain, self.provider_factory(), media)
            self.session = session  # stop can reach even an unfinished handshake
            await session.start()
            self._ticket = secrets.token_urlsafe(32)
            self._ticket_deadline = asyncio.get_running_loop().time() + 10
            async def expire_ticket():
                await asyncio.sleep(10)
                if self.session is session and self._ticket is not None:
                    self._ticket = None
                    await session.close("media_ticket_expired")
            if self._ticket_task:
                self._ticket_task.cancel()
            self._ticket_task = asyncio.create_task(expire_ticket())
            return web.json_response({"session_id": session.session_id, "robot_id": self.domain.robot_id,
                "ticket": self._ticket, "format": media.format.as_dict(),
                "barge_in": self.domain.barge_in, "physical_admission": False})
        except web.HTTPException:
            raise
        except Exception:  # noqa: BLE001 — fail closed at the external protocol boundary
            # Never disclose provider endpoint/authentication exceptions.
            raise web.HTTPBadGateway(text="provider session unavailable") from None
        finally:
            self._creating = False

    async def _status(self, request):
        from aiohttp import web
        return web.json_response({"robot_id": self.domain.robot_id,
            "ready": bool(self.session and self.session.ready), "stopped": self.domain.runtime.stopped,
            "generation": self.domain.runtime.cancellation_token,
            "tools": list(self.domain.tools), "barge_in": self.domain.barge_in})

    async def _stop(self, request):
        from aiohttp import web
        if self.session and not self.session.closed:
            receipt = await self.session.interrupt("operator_stop", send_cancel=False, force_stop=True)
        else:
            receipt = await self.domain.stop()
        return web.json_response(receipt)

    async def _reset(self, request):
        from aiohttp import web
        # Explicit authenticated operator route, absent from the model tool list.
        return web.json_response(await asyncio.to_thread(self.domain.runtime.reset_stop))

    async def _disconnect(self, request):
        from aiohttp import web
        receipt = {"ok": True, "session_closed": True, "action_pending": self.domain.action_pending}
        if self.session:
            receipt = await self.session.close("operator_disconnect")
        if self._socket:
            await self._socket.close()
        self._ticket = None
        return web.json_response(receipt)

    async def _media(self, request):
        from aiohttp import WSMsgType, web
        # A short-lived single-use ticket binds the WebSocket to the previously
        # authenticated HTTP session. Access logs are disabled; no provider key
        # is ever sent to the browser. Cross-origin browser sockets are refused.
        valid = (self._origin_valid(request) and self._ticket is not None
                 and hmac.compare_digest(request.query.get("ticket", ""), self._ticket)
                 and asyncio.get_running_loop().time() <= self._ticket_deadline
                 and self.session is not None and self.session.ready)
        if not valid:
            raise web.HTTPForbidden(text="invalid media session")
        self._ticket = None
        session = self.session
        ws = web.WebSocketResponse(max_msg_size=16384, heartbeat=10)
        await ws.prepare(request)
        self._socket = ws

        async def output():
            while not session.closed:
                event = await session.media.receive()
                event["session_id"] = session.session_id
                await asyncio.wait_for(ws.send_json(event), 2)

        async def incoming():
            async for message in ws:
                if message.type != WSMsgType.TEXT:
                    raise ValueError("expected media JSON")
                event = json.loads(message.data)
                if not isinstance(event, dict) or event.get("session_id") != session.session_id:
                    raise ValueError("media session mismatch")
                if event.get("type") == "audio" and set(event) == {"type", "session_id", "sequence", "audio"}:
                    session.media.feed(event["sequence"], decode_pcm(event["audio"], maximum_bytes=4800))
                elif event.get("type") == "text" and set(event) == {"type", "session_id", "text"}:
                    await session.text(event["text"])
                elif event.get("type") == "interrupt" and set(event) == {"type", "session_id"}:
                    await session.interrupt()
                else:
                    raise ValueError("unknown media command")

        tasks = [asyncio.create_task(output()), asyncio.create_task(incoming()),
                 asyncio.create_task(session.done.wait())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await session.close("browser_media_disconnected")
            await ws.close()
            self._socket = None
        return ws

    async def close(self):
        if self._ticket_task:
            self._ticket_task.cancel()
            await asyncio.gather(self._ticket_task, return_exceptions=True)
        if self.session:
            await self.session.close("gateway_shutdown")
        if self._socket:
            await self._socket.close()
        if self._runner:
            await self._runner.cleanup()
        return await self.domain.close()
