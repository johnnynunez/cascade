"""Authenticated, loopback-only HTTP/media gateway; optional aiohttp imports."""
from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import re
import secrets
import time
from importlib.resources import files

from .media import QueueMediaIO, decode_pcm
from .session import ConversationSession
from .lifecycle import close_stage
from ..lifecycle import teardown_receipt


class ConversationGateway:
    def __init__(self, domain, provider_factory, *, token=None, activation=None, probes=None):
        self.domain, self.provider_factory = domain, provider_factory
        self.activation = activation
        # Split deployments only (B51): () -> {"ready": bool, "reason": ...}
        # for the remote robot. None keeps the original route table.
        self.probes = probes
        self._capture_ready = None
        self._activation_task = self._episode_task = None
        self._activating = False
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
        self._closing = False
        self._close_lock = asyncio.Lock()
        self.closure_receipt = None

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
        app.router.add_post("/api/robot/start", self._start_robot)
        app.router.add_get("/api/status", self._status)
        app.router.add_get("/api/media", self._media)
        if self.probes is not None:
            # Unauthenticated supervisor probes: no token, session, tool or
            # generation detail. Everything under /api/ stays authenticated.
            app.router.add_get("/healthz", self._healthz)
            app.router.add_get("/readyz", self._readyz)
        return app

    async def start(self, *, host="127.0.0.1", port=8780):
        from aiohttp import web
        if self._closing or self._runner is not None:
            raise ValueError("gateway already started or closed")
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
        if self._closing:
            raise web.HTTPServiceUnavailable(text="gateway closing")
        if self._creating or self._activating or (self.session is not None and (not self.session.closed
                or not self.session.closure_receipt or not self.session.closure_receipt.get("ok"))):
            raise web.HTTPConflict(text="session already exists")
        if self.activation is not None and self.activation.status()["attempted"]:
            try:
                self.activation.check_live()
            except ValueError:
                raise web.HTTPConflict(text="Robot session ended. Start a new service run.") from None
        # Reserve before the first await: two slow HTTP bodies must not both
        # pass admission and overwrite the sole owned provider session.
        self._creating = True
        try:
            body = await request.json()
            if self._closing:
                raise web.HTTPServiceUnavailable(text="gateway closing")
            if body != {"robot_id": self.domain.robot_id}:
                raise web.HTTPBadRequest(text="robot identity mismatch")
            media = QueueMediaIO()
            session = ConversationSession(self.domain, self.provider_factory(), media)
            self.session = session  # stop can reach even an unfinished handshake
            await session.start()
            if self.activation is not None and self.activation.status()["attempted"]:
                try:
                    self.activation.check_live()
                except BaseException:
                    await session.close("robot_episode_expired_during_connect")
                    raise
            if self._closing or session.closed:
                await session.close("gateway_shutdown")
                raise web.HTTPServiceUnavailable(text="gateway closing")
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

    async def _healthz(self, request):
        from aiohttp import web
        return web.json_response({"alive": True})

    async def _readyz(self, request):
        from aiohttp import web
        robot = await asyncio.to_thread(self.probes)
        # A closing gateway refuses new sessions, so it is never ready.
        ready = robot.get("ready") is True and not self._closing
        return web.json_response({"ready": ready, "robot": robot}, status=200 if ready else 503)

    async def _status(self, request):
        from aiohttp import web
        return web.json_response({"robot_id": self.domain.robot_id,
            "ready": bool(self.session and self.session.ready), "stopped": self.domain.runtime.stopped,
            "generation": self.domain.runtime.cancellation_token,
            "tools": list(self.domain.tools), "barge_in": self.domain.barge_in,
            **({"robot_episode": self.activation.status()} if self.activation is not None else {})})

    async def _start_robot(self, request):
        from aiohttp import web
        if self.activation is None:
            raise web.HTTPNotFound(text="explicit robot activation is not configured")
        body = await request.json()
        session, socket = self.session, self._socket
        if (type(body) is not dict or set(body) != {"session_id", "generation", "capture_id"}
                or type(body["generation"]) is not int or session is None
                or not session.ready or session.closed or self._closing or self._activating
                or socket is None or socket.closed
                or body["session_id"] != session.session_id
                or body["generation"] != self.domain.runtime.cancellation_token
                or self._capture_ready != (session, socket, body["capture_id"])
                or self.activation.status()["attempted"]):
            raise web.HTTPConflict(text="robot start requires the current prepared microphone session")
        self._activating = True
        self._capture_ready = None
        # A separate finite activation request, not a replacement hand lifetime.
        deadline = time.monotonic() + 10.
        async def activate():
            try:
                receipt = await session.close("operator_start_robot_session")
                generation = body["generation"] + 1
                if (receipt.get("ok") is not True
                        or receipt.get("stop", {}).get("generation") != generation
                        or self.domain.runtime.cancellation_token != generation
                        or self._closing or self.session is not session or time.monotonic() >= deadline):
                    raise ValueError("robot start closure was refused, superseded or late")
                result = await asyncio.to_thread(self.activation.activate, generation, deadline=deadline)
                if self._closing or time.monotonic() >= deadline:
                    raise ValueError("robot start returned after cancellation or deadline")
                self.activation.check_live()
                self._episode_task = asyncio.create_task(self._watch_episode())
                return {**result, "reconnect_required": True, "activation_deadline_monotonic_s": deadline}
            except BaseException:
                self.activation.revoke()
                await self.domain.stop()
                raise
            finally:
                self._activating = False
        self._activation_task = asyncio.create_task(activate())
        self._activation_task.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        try:
            result = await asyncio.wait_for(asyncio.shield(self._activation_task),
                                            max(0., deadline-time.monotonic()))
            return web.json_response(result)
        except (Exception, asyncio.CancelledError):
            # The task/constructor remains owned; a late result cannot publish.
            self.activation.revoke()
            await self.domain.stop()
            raise web.HTTPConflict(text="robot activation failed; no new episode was granted") from None

    async def _watch_episode(self):
        while not self._closing:
            try:
                self.activation.check_live()
            except Exception:
                await self.domain.stop()
                if self.session and not self.session.closed:
                    try:
                        await self.session.media.emit({"type": "robot_episode_ended"})
                    except (ValueError, asyncio.QueueFull):
                        pass
                    await self.session.close("robot_episode_expired_or_failed")
                await asyncio.to_thread(self.activation.close)
                return
            await asyncio.sleep(.05)

    async def _stop(self, request):
        from aiohttp import web
        if self.session and not self.session.closed:
            receipt = await self.session.interrupt("operator_stop", send_cancel=False, force_stop=True)
        else:
            receipt = await self.domain.stop()
        return web.json_response(receipt)

    async def _reset(self, request):
        from aiohttp import web
        if self._closing:
            raise web.HTTPServiceUnavailable(text="gateway closing")
        if self._activating:
            raise web.HTTPConflict(text="explicit robot start is already in progress")
        # Explicit authenticated operator route, absent from the model tool list.
        try:
            body = await request.json()
        except (ValueError, UnicodeError) as exc:
            raise web.HTTPBadRequest(text="reset requires a JSON generation") from exc
        if (type(body) is not dict or set(body) != {"generation"}
                or type(body["generation"]) is not int or body["generation"] < 0):
            raise web.HTTPBadRequest(text="reset requires an exact nonnegative integer generation")
        if self._closing:
            raise web.HTTPServiceUnavailable(text="gateway closing")
        if self._activating:
            raise web.HTTPConflict(text="explicit robot start is already in progress")
        return web.json_response(await asyncio.to_thread(
            self.domain.runtime.reset_stop, expected_generation=body["generation"]))

    async def _disconnect(self, request):
        from aiohttp import web
        activation_stop = None
        if self._activating:
            # Start has already closed the old provider session. Closing that
            # session again is idempotent and cannot cancel its constructor.
            self.activation.revoke()  # Revoke before the first await.
            activation_stop = await self.domain.stop()
        receipt = {"ok": True, "session_closed": True, "action_pending": self.domain.action_pending}
        if self.session:
            receipt = await self.session.close("operator_disconnect")
        if self._socket:
            await self._socket.close()
        self._ticket = None
        if activation_stop is not None:
            receipt = {**receipt, "ok": receipt.get("ok") is True and activation_stop.get("ok") is True,
                       "activation_cancelled": True, "activation_stop": activation_stop}
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
                    if self.activation is not None:
                        self.activation.check_live()
                    session.media.feed(event["sequence"], decode_pcm(event["audio"], maximum_bytes=4800))
                elif event.get("type") == "text" and set(event) == {"type", "session_id", "text"}:
                    if self.activation is not None:
                        self.activation.check_live()
                    await session.text(event["text"])
                elif (self.activation is not None and event.get("type") == "microphone_prepared"
                      and set(event) == {"type", "session_id", "capture_id"}
                      and type(event["capture_id"]) is str
                      and re.fullmatch(r"[0-9a-f]{32}", event["capture_id"])
                      and not self.activation.status()["attempted"]):
                    self._capture_ready = (session, ws, event["capture_id"])
                    await session.media.emit({"type": "microphone_prepared", "capture_id": event["capture_id"]})
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
            if self._socket is ws:
                self._socket = None
            if self._capture_ready and self._capture_ready[1] is ws:
                self._capture_ready = None
        return ws

    async def close(self):
        async with self._close_lock:
            if self.closure_receipt is not None:
                return self.closure_receipt
            self._closing = True
            self._ticket = None
            if self._activating:
                self.activation.revoke()
            stages = []
            if self._episode_task:
                self._episode_task.cancel()
                async def drain_episode():
                    await asyncio.gather(self._episode_task, return_exceptions=True)
                stages.append(await close_stage("episode_watch", drain_episode))
            if self._activation_task:
                async def drain_activation():
                    await asyncio.wait_for(asyncio.shield(self._activation_task), 5.)
                stages.append(await close_stage("activation", drain_activation))
            if self._ticket_task:
                self._ticket_task.cancel()
                async def drain_ticket():
                    await asyncio.gather(self._ticket_task, return_exceptions=True)
                stages.append(await close_stage("ticket", drain_ticket))
            if self.session:
                stages.append(await close_stage("session", lambda: self.session.close("gateway_shutdown")))
            if self._socket:
                socket = self._socket
                async def close_socket():
                    # aiohttp's bool means "newly closed", not a close receipt.
                    await socket.close()
                stages.append(await close_stage("socket", close_socket))
            if self._runner:
                stages.append(await close_stage("http", self._runner.cleanup))
            stages.append(await close_stage("domain", self.domain.close))
            domain_receipt = stages[-1].get("result", {})
            self.closure_receipt = {**domain_receipt, **teardown_receipt(stages)}
            return self.closure_receipt
