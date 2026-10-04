"""Real HTTP/WebSocket shutdown ordering; no model or physical acceptance."""
import asyncio
import base64

import pytest
from test_conversation_protocol import aiohttp, created, pcm_fixture, rig


@pytest.mark.parametrize("stop_first", [False, True], ids=["websocket-first", "http-stop-first"])
@pytest.mark.parametrize("response_started", [False, True], ids=["pending-input", "playing-response"])
def test_terminal_close_revokes_before_provider_release(monkeypatch, stop_first, response_started):
    async def scenario():
        async with rig(gateway=True) as (robot, runtime, _, wire, _, _, gate, origin):
            release, closing = asyncio.Event(), asyncio.Event()
            provider_closes = []
            async with aiohttp.ClientSession(headers={"Authorization": "Bearer " + gate.token}) as client:
                async with client.post(origin + "/api/session", json={"robot_id": "fixture"}) as response:
                    assert response.status == 200
                    binding = await response.json()
                session = gate.session
                original_close = session.provider.close

                async def blocked_close():
                    provider_closes.append(True)
                    closing.set()
                    await release.wait()
                    return await original_close()

                monkeypatch.setattr(session.provider, "close", blocked_close)
                ws = await client.ws_connect(origin + "/api/media?ticket=" + binding["ticket"])
                try:
                    if response_started:
                        await created(wire)
                        await wire.emit({"type": "response.output_audio.delta", "response_id": "r1",
                                         "delta": base64.b64encode(pcm_fixture()).decode()})
                        assert (await ws.receive_json(timeout=3))["type"] == "audio"
                        assert session.responses["r1"].valid
                    else:
                        await ws.send_json({"type": "text", "session_id": session.session_id,
                                            "text": "pending fixture input"})
                        await wire.next("response.create")
                        assert session._pending is not None
                    if stop_first:
                        async with client.post(origin + "/api/stop") as response:
                            assert response.status == 200 and (await response.json())["ok"]
                        assert session.authority_revoked and session._pending is None
                    await ws.close()
                    await asyncio.wait_for(closing.wait(), 3)
                    # Provider teardown has not returned: terminal authority must
                    # already be gone, regardless of the browser/HTTP ordering.
                    assert not release.is_set() and not session.done.is_set()
                    assert session.closed and not session.ready and session.authority_revoked
                    assert session._pending is None
                    assert all(not context.valid for context in session.responses.values())
                    if not stop_first:
                        async with client.post(origin + "/api/stop") as response:
                            assert response.status == 200 and (await response.json())["ok"]
                    assert runtime.stopped and robot.stopped.is_set()
                    existing = tuple(session.responses)
                    # An event already delivered to a caller after close cannot
                    # reacquire a context, dispatch a tool or revive pending input.
                    await session.handle({"type": "response.created", "response": {"id": "late"}})
                    await session.handle({"type": "response.function_call_arguments.done",
                        "response_id": "r1", "call_id": "late", "name": "robot_tool_0",
                        "output_index": 0, "arguments": '{"vx":0.1}'})
                    assert tuple(session.responses) == existing and session._pending is None
                    assert robot.calls == []
                finally:
                    release.set()
                    await ws.close()
                await asyncio.wait_for(session.done.wait(), 3)
                original_receipt = session.closure_receipt
                assert original_receipt["ok"] and original_receipt["reason"] == "browser_media_disconnected"
                assert await session.close("repeated") is original_receipt
                assert session.authority_revoked and session._pending is None
                assert provider_closes == [True]
    asyncio.run(scenario())


def test_terminal_authority_is_revoked_before_first_stop_await_even_when_close_fails(monkeypatch):
    async def scenario():
        async with rig() as (_, _, domain, wire, session, *_):
            await session.text("pending fixture input")
            await wire.next("response.create")
            assert session._pending is not None
            original_stop = domain.stop
            observed = []

            async def failing_stop():
                observed.append((session.closed, session.ready, session.authority_revoked, session._pending))
                await original_stop()
                raise OSError("fixture stop-record failure")

            monkeypatch.setattr(domain, "stop", failing_stop)
            try:
                receipt = await session.close("fixture_failure")
                assert observed == [(True, False, True, None)]
                assert receipt["ok"] is False and receipt["complete"] is False
                assert session.done.is_set() and session.authority_revoked and session._pending is None
                assert await session.close("again") is receipt
                assert len(observed) == 1
            finally:
                monkeypatch.setattr(domain, "stop", original_stop)
    asyncio.run(scenario())
