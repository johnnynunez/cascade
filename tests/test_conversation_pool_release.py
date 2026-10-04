"""Actual loopback reconnect races; no speech models or physical robot."""
import asyncio
from contextlib import asynccontextmanager

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web

from cascade.conversation.domain import ConversationDomain
from cascade.conversation.gateway import ConversationGateway
from cascade.conversation.provider import RealtimeConfig, RealtimeWebSocket
from cascade.conversation.provider_pool import validate_pool
from cascade.robotics.runtime import RobotRuntime
from conftest import loopback_host
from test_conversation_protocol import Robot


class Pool:
    """Same release ordering as pinned HF: transport closes before drain."""
    def __init__(self):
        self.session = None
        self.state = "idle"
        self.connections = 0
        self.rejected = 0
        self.draining = asyncio.Event()
        self.release = asyncio.Event()
        self.observed_draining = asyncio.Event()
        self.tasks = []
        self.change = lambda snapshot: snapshot

    async def socket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.connections += 1
        if self.session is not None:
            self.rejected += 1
            await ws.send_json({"type": "error", "error": {"type": "session_limit_reached"}})
            await ws.close()
            return ws
        self.session = "session-" + str(self.connections)
        self.state = "active"
        await ws.send_json({"type": "session.created", "session": {"id": self.session}})
        async for message in ws:
            data = message.json()
            if data["type"] == "session.update":
                await ws.send_json({"type": "session.updated", "session": data["session"]})
        self.state = "draining"
        self.draining.set()
        async def drain():
            await self.release.wait()
            self.session, self.state = None, "idle"
        self.tasks.append(asyncio.create_task(drain()))
        return ws

    async def snapshot(self, request):
        unit = {"index": 0, "state": self.state, "session_id": self.session}
        if self.state == "draining":
            unit["draining_for_s"] = 0.1
            self.observed_draining.set()
        result = self.change({"size": 1, "in_use": int(self.session is not None), "units": [unit]})
        return result if isinstance(result, web.StreamResponse) else web.json_response(result)


@asynccontextmanager
async def environment(contract, *, timeout=1):
    pool = Pool()
    app = web.Application()
    app.router.add_get("/v1/realtime", pool.socket)
    app.router.add_get("/v1/pool", pool.snapshot)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, loopback_host(), 0).start()
    url = f"ws://{loopback_host()}:{runner.addresses[0][1]}/v1/realtime"
    runtime = RobotRuntime({"body": Robot()})
    domain = ConversationDomain(runtime, robot_id="fixture", allow_tools=("body.read",), allow_motion=False)
    providers = []
    def factory():
        provider = RealtimeWebSocket(RealtimeConfig(url, release_contract=contract, io_timeout_s=timeout))
        providers.append(provider)
        return provider
    gate = ConversationGateway(domain, factory)
    origin = await gate.start(port=0)
    try:
        async with aiohttp.ClientSession(headers={"Authorization": "Bearer " + gate.token}) as client:
            yield pool, gate, client, origin, providers
    finally:
        pool.release.set()
        await gate.close()
        await asyncio.gather(*pool.tasks)
        await runner.cleanup()
        runtime.close()


@pytest.mark.parametrize("contract", [None, "hf_pool"])
def test_real_delete_reconnect_waits_for_actual_release_only_when_requested(contract):
    async def scenario():
        async with environment(contract) as (pool, gate, client, origin, providers):
            # One absolute test/action window; no repeated POST or new wait budget.
            deadline = asyncio.get_running_loop().time() + 3
            async def create():
                return await client.post(origin + "/api/session", json={"robot_id": "fixture"})
            assert (await create()).status == 200
            closing = asyncio.create_task(client.delete(origin + "/api/session"))
            await asyncio.wait_for(pool.draining.wait(), deadline - asyncio.get_running_loop().time())
            if contract is None:
                assert (await (await closing).json())["ok"]
                response = await create()
                assert response.status == 502
                assert await response.text() == "provider session unavailable"
                assert pool.rejected == 1
            else:
                await asyncio.wait_for(pool.observed_draining.wait(), deadline - asyncio.get_running_loop().time())
                assert not closing.done() and not gate.session.closure_receipt
                pool.release.set()  # Release event, not an elapsed sleep/phase choice.
                receipt = await (await closing).json()
                assert receipt["ok"] and pool.state == "idle"
                # Original browser sequence: reset and exactly one new connect.
                reset = await client.post(origin + "/api/reset", json={"generation": gate.domain.runtime.cancellation_token})
                assert (await reset.json())["ok"]
                assert (await create()).status == 200
                assert pool.connections == 2 and pool.rejected == 0
                release = next(s for s in providers[0]._closure_receipt["stages"] if s["stage"] == "pool_release")
                assert release["ok"]
            assert asyncio.get_running_loop().time() < deadline
    asyncio.run(scenario())


def test_release_timeout_is_sticky_closes_http_and_blocks_new_provider():
    async def scenario():
        async with environment("hf_pool", timeout=.12) as (pool, gate, client, origin, providers):
            assert (await client.post(origin + "/api/session", json={"robot_id": "fixture"})).status == 200
            receipt = await (await client.delete(origin + "/api/session")).json()
            assert not receipt["ok"] and pool.state == "draining"
            assert providers[0]._client is None
            assert (await client.post(origin + "/api/session", json={"robot_id": "fixture"})).status == 409
            old_deadline = providers[0]._pool_close_deadline
            pool.release.set()
            assert (await providers[0].close())["ok"] is False
            assert providers[0]._pool_close_deadline == old_deadline
            assert pool.connections == 1
    asyncio.run(scenario())


@pytest.mark.parametrize("change", [
    lambda x: {**x, "units": [{**x["units"][0], "state": "stuck", "stuck_for_s": 0.1}]},
    lambda x: {**x, "units": [{**x["units"][0], "index": 1}]},
    lambda x: {**x, "units": [{**x["units"][0], "session_id": "replacement"}]},
    lambda x: {"size": 2, "in_use": 1, "units": [
        {"index": 0, "state": "idle", "session_id": None}, {**x["units"][0], "index": 1}]},
])
def test_close_refuses_stuck_replaced_or_relocated_session(change):
    async def scenario():
        async with environment("hf_pool") as (pool, gate, client, origin, providers):
            assert (await client.post(origin + "/api/session", json={"robot_id": "fixture"})).status == 200
            pool.change = change
            receipt = await (await client.delete(origin + "/api/session")).json()
            assert receipt["ok"] is False
            assert providers[0]._client is None and pool.state == "draining"
            assert (await client.post(origin + "/api/session", json={"robot_id": "fixture"})).status == 409
    asyncio.run(scenario())


@pytest.mark.parametrize("response", [
    lambda: web.Response(status=302, headers={"Location": "http://unrelated.invalid/v1/pool"}),
    lambda: web.Response(text="provider unavailable", content_type="text/plain"),
    lambda: web.Response(text='{"size":1,"size":0,"units":[],"in_use":0}', content_type="application/json"),
    lambda: web.Response(body=b" "*262145, content_type="application/json"),
])
def test_pool_http_redirect_nonjson_duplicate_or_oversized_body_is_not_release(response):
    async def scenario():
        async with environment("hf_pool") as (pool, gate, client, origin, providers):
            pool.change = lambda value: response()
            assert (await client.post(origin + "/api/session", json={"robot_id": "fixture"})).status == 502
            assert gate.session.closure_receipt["ok"] is False
            assert providers[0]._client is None
    asyncio.run(scenario())


@pytest.mark.parametrize("change", [
    lambda x: {**x, "in_use": 0},
    lambda x: {**x, "size": True},
    lambda x: {**x, "units": []},
    lambda x: {**x, "size": 2, "in_use": 2, "units": x["units"] * 2},
    lambda x: {**x, "units": [{**x["units"][0], "session_id": "other"}]},
    lambda x: {**x, "units": [{**x["units"][0], "state": "idle"}]},
])
def test_handshake_pool_mismatch_never_certifies_release(change):
    async def scenario():
        async with environment("hf_pool") as (pool, gate, client, origin, providers):
            pool.change = change
            assert (await client.post(origin + "/api/session", json={"robot_id": "fixture"})).status == 502
            assert gate.session.closure_receipt["ok"] is False
            assert providers[0]._client is None
            assert pool.connections == 1
    asyncio.run(scenario())


@pytest.mark.parametrize("state,extra", [("draining", {"draining_for_s": float("nan")}),
    ("draining", {"draining_for_s": True}), ("stuck", {"draining_for_s": 1}),
    ("unknown", {}), ("active", {"unexpected": 1})])
def test_pool_schema_refuses_ambiguous_state(state, extra):
    snapshot = {"size": 1, "in_use": 1, "units": [{"index": 0, "state": state, "session_id": "s", **extra}]}
    with pytest.raises(ValueError):
        validate_pool(snapshot)


def test_release_contract_is_explicit_and_has_no_alternate_authority(tmp_path):
    import json
    from cascade.apps.conversation import parser
    from cascade.conversation.service import configuration
    url = "ws://localhost:1234/v1/realtime"
    assert RealtimeConfig(url).release_contract is None
    for bad in ("automatic", True, 1, [], {}):
        with pytest.raises(ValueError):
            RealtimeConfig(url, release_contract=bad)
    for suffix in ("?token=x", "/other"):
        with pytest.raises(ValueError):
            RealtimeConfig(url + suffix, release_contract="hf_pool")
    config = tmp_path / "service.json"
    config.write_text(json.dumps({"version": 1, "provider_url": url, "run_root": "runs",
                                  "provider_release_contract": "hf_pool"}))
    assert configuration(parser().parse_args(["--config", str(config)]))["provider_release_contract"] == "hf_pool"


def test_opt_in_websocket_upgrade_redirect_is_refused_before_following():
    async def scenario():
        calls = []
        async def target(request):
            calls.append(request.path)
            return web.Response(status=500)
        foreign = web.Application()
        foreign.router.add_get('/other', target)
        foreign_runner = web.AppRunner(foreign)
        await foreign_runner.setup()
        await web.TCPSite(foreign_runner, loopback_host(), 0).start()
        async def redirect(request):
            raise web.HTTPFound(f'http://{loopback_host()}:{foreign_runner.addresses[0][1]}/other')
        original = web.Application()
        original.router.add_get('/v1/realtime', redirect)
        runner = web.AppRunner(original)
        await runner.setup()
        await web.TCPSite(runner, loopback_host(), 0).start()
        provider = RealtimeWebSocket(RealtimeConfig(
            f'ws://{loopback_host()}:{runner.addresses[0][1]}/v1/realtime', release_contract='hf_pool'))
        try:
            with pytest.raises(ValueError, match='must not redirect'):
                await provider.connect({})
            assert calls == [] and provider._client is None
            assert (await provider.close())['ok'] is False
        finally:
            await runner.cleanup()
            await foreign_runner.cleanup()
    asyncio.run(scenario())
