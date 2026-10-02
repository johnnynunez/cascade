"""Real loopback HTTP/WebSocket/audio fixtures; no speech inference or robot physics."""
import asyncio
import base64
import copy
import io
import json
import math
import struct
import threading
import wave
from contextlib import asynccontextmanager

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web

from cascade.conversation.domain import ConversationDomain
from cascade.conversation.gateway import ConversationGateway
from cascade.conversation.media import QueueMediaIO
from cascade.conversation.provider import RealtimeConfig, RealtimeWebSocket
from cascade.conversation.session import ConversationSession
from cascade.robotics.contracts import ResourceDescriptor, ToolDescriptor
from cascade.robotics.runtime import RobotRuntime


class Robot:
    domain_id = "body"

    def __init__(self):
        self.resources = (ResourceDescriptor("body/base", "base", "fixture", synthetic=True,
            admission="software_only", controller_id="fixture/controller", writer_id="body"),)
        schema = {"type": "object", "properties": {"vx": {"type": "number", "minimum": -.3, "maximum": .3}},
                  "required": ["vx"], "additionalProperties": False}
        self.tool_descriptors = (
            ToolDescriptor("body.walk_velocity", "Synthetic bounded velocity", schema, "body", "walk_velocity",
                           effect="motion", writes=("body/base",)),
            ToolDescriptor("body.read", "Synthetic observation", {"type": "object", "properties": {},
                           "additionalProperties": False}, "body", "read"))
        self.calls = []
        self.entered, self.stopped = threading.Event(), threading.Event()
        self.block = None

    def execute(self, name, args):
        self.calls.append((name, args))
        self.entered.set()
        if self.block:
            assert self.block.wait(5), "test did not release its blocked action"
        return {"ok": True, "synthetic": True, "postcondition": {"status": "unverified"}}

    def stop(self):
        self.stopped.set()
        return {"ok": True}

    def reset_stop(self):
        return {"ok": True}

    def close(self):
        return {"ok": True}


class Wire:
    def __init__(self):
        self.messages = asyncio.Queue()
        self.ws = None
        self.sessions = []
        self.ack = asyncio.Event()
        self.ack.set()
        self.updated = asyncio.Event()
        self.bad_format = False

    async def handle(self, request):
        self.ws = web.WebSocketResponse()
        await self.ws.prepare(request)
        ws = self.ws
        await ws.send_json({"type": "session.created", "session": {"id": "fixture"}})
        async for message in ws:
            if message.type != aiohttp.WSMsgType.TEXT:
                break
            event = json.loads(message.data)
            if event["type"] == "session.update":
                self.sessions.append(event["session"])
                self.updated.set()
                await self.ack.wait()
                effective = copy.deepcopy(event["session"])
                if self.bad_format:
                    effective["audio"]["output"]["format"]["rate"] = 16000
                await ws.send_json({"type": "session.updated", "session": effective})
            else:
                await self.messages.put(event)
        return ws

    async def emit(self, event):
        await self.ws.send_json(event)

    async def next(self, kind):
        while True:
            event = await asyncio.wait_for(self.messages.get(), 3)
            if event["type"] == kind:
                return event


@asynccontextmanager
async def rig(*, gateway=False, **options):
    robot = Robot()
    runtime = RobotRuntime({"body": robot})
    domain = ConversationDomain(runtime, robot_id="fixture", allow_tools=("body.walk_velocity", "body.read"),
                                allow_motion=True, **options)
    wire = Wire()
    app = web.Application()
    app.router.add_get("/v1/realtime", wire.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    url = f"ws://127.0.0.1:{runner.addresses[0][1]}/v1/realtime"
    factory = lambda: RealtimeWebSocket(RealtimeConfig(url, connect_timeout_s=3))
    media = QueueMediaIO()
    session = ConversationSession(domain, factory(), media)
    gate = ConversationGateway(domain, factory)
    try:
        if gateway:
            origin = await gate.start(port=0)
        else:
            origin = None
            await session.start()
        yield robot, runtime, domain, wire, session, media, gate, origin
    finally:
        if robot.block:
            robot.block.set()
        if gateway:
            await gate.close()
        else:
            await session.close()
            await domain.close()
        runtime.close()
        wire.ack.set()
        await runner.cleanup()


async def created(wire, rid="r1"):
    await wire.emit({"type": "response.created", "response": {"id": rid, "status": "in_progress"}})


async def call(wire, *, rid="r1", cid="c1", alias="robot_tool_0", args=None):
    await wire.emit({"type": "response.function_call_arguments.done", "response_id": rid,
        "call_id": cid, "name": alias, "output_index": 0, "arguments": json.dumps({"vx": .1} if args is None else args)})


async def terminal(wire, *, rid="r1", cid="c1", status="completed"):
    await wire.emit({"type": "response.done", "response": {"id": rid, "status": status,
        "output": [{"type": "function_call", "call_id": cid}]}})


def pcm_fixture():
    # Actual WAV encoding/decoding establishes format, not just arbitrary bytes.
    pcm = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / 24000))) for i in range(960))
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000); wav.writeframes(pcm)
    with wave.open(io.BytesIO(out.getvalue()), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 24000)
        return wav.readframes(960)


def test_browser_gateway_roundtrip_audio_and_completed_typed_tool():
    async def scenario():
        async with rig(gateway=True) as (robot, runtime, _domain, wire, _, _, gate, origin):  # noqa: SIM117 — fixture bindings create client credentials
            async with aiohttp.ClientSession(headers={"Authorization": "Bearer " + gate.token}) as client:
                async with client.get(origin + "/") as response:
                    assert response.status == 200 and "Talk to your robot" in await response.text()
                async with client.post(origin + "/api/session", json={"robot_id": "fixture"}) as response:
                    binding = await response.json()
                    assert response.status == 200
                async with client.ws_connect(origin + "/api/media?ticket=" + binding["ticket"]) as ws:
                    pcm = pcm_fixture()
                    await ws.send_json({"type": "audio", "session_id": binding["session_id"],
                                        "sequence": 0, "audio": base64.b64encode(pcm).decode()})
                    upstream = await wire.next("input_audio_buffer.append")
                    assert base64.b64decode(upstream["audio"]) == pcm
                    await created(wire)
                    await wire.emit({"type": "response.output_audio.delta", "response_id": "r1",
                                     "delta": base64.b64encode(pcm).decode()})
                    playback = await ws.receive_json(timeout=3)
                    assert playback["type"] == "audio" and playback["session_id"] == binding["session_id"]
                    assert base64.b64decode(playback["audio"]) == pcm
                    await call(wire)
                    # Round-trip barrier through the same ordered server stream.
                    await wire.emit({"type": "response.output_audio_transcript.done", "response_id": "r1", "transcript": "staged"})
                    assert (await ws.receive_json(timeout=3))["text"] == "staged"
                    assert robot.calls == []
                    await terminal(wire)
                    result = await wire.next("conversation.item.create")
                    assert result["item"]["type"] == "function_call_output"
                    assert json.loads(result["item"]["output"])["postcondition"]["status"] == "unverified"
                    assert robot.calls == [("walk_velocity", {"vx": .1})]
                    assert runtime.unverified_actions()
                await asyncio.wait_for(gate.session.done.wait(), 3)
                assert robot.stopped.is_set()
    asyncio.run(scenario())


@pytest.mark.parametrize("status", ["cancelled", "failed", "incomplete"])
def test_noncompleted_response_never_executes_staged_tool(status):
    async def scenario():
        async with rig() as (robot, _, _, wire, session, media, *_):
            await created(wire); await call(wire); await terminal(wire, status=status)
            await created(wire, "barrier")
            await wire.emit({"type": "response.output_audio_transcript.done", "response_id": "barrier", "transcript": "done"})
            assert (await asyncio.wait_for(media.receive(), 3))["text"] == "done"
            assert robot.calls == [] and not session.closed
    asyncio.run(scenario())


def test_stop_reset_between_response_birth_and_tool_dispatch_refuses_old_generation():
    async def scenario():
        async with rig() as (robot, runtime, _, wire, _, media, *_):
            await created(wire)
            await wire.emit({"type": "response.output_audio_transcript.done", "response_id": "r1", "transcript": "born"})
            await asyncio.wait_for(media.receive(), 3)
            assert runtime.stop()["ok"] and runtime.reset_stop()["ok"]
            await call(wire); await terminal(wire)
            result = await wire.next("conversation.item.create")
            assert "stale execution generation" in json.loads(result["item"]["output"])["error"]
            assert robot.calls == []
    asyncio.run(scenario())


def test_expired_context_does_not_execute():
    async def scenario():
        async with rig(intent_timeout_s=.01) as (robot, _, _, wire, _, media, *_):
            await created(wire)
            await wire.emit({"type": "response.output_audio_transcript.done", "response_id": "r1", "transcript": "born"})
            await asyncio.wait_for(media.receive(), 3)
            await asyncio.sleep(.02)
            await call(wire); await terminal(wire)
            result = await wire.next("conversation.item.create")
            assert "expired" in json.loads(result["item"]["output"])["error"]
            assert robot.calls == []
    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["duplicate", "unknown_tool", "bad_args", "wrong_origin"])
def test_protocol_fault_closes_and_stops_without_action(fault):
    async def scenario():
        async with rig() as (robot, _, _, wire, session, *_):
            await created(wire)
            if fault == "duplicate":
                await call(wire); await call(wire)
            elif fault == "unknown_tool":
                await call(wire, alias="reset_stop")
            elif fault == "bad_args":
                await call(wire, args={"vx": 100})
            else:
                await call(wire, rid="old_session")
            await asyncio.wait_for(session.done.wait(), 3)
            assert robot.calls == [] and robot.stopped.is_set()
    asyncio.run(scenario())


def test_barge_in_flushes_and_invalidates_partial_tool_without_stopping_idle_first_speech():
    async def scenario():
        async with rig() as (robot, runtime, _, wire, _, media, *_):
            await wire.emit({"type": "input_audio_buffer.speech_started"})
            assert (await asyncio.wait_for(media.receive(), 3))["type"] == "flush"
            assert not runtime.stopped
            await created(wire); await call(wire)
            await wire.emit({"type": "input_audio_buffer.speech_started"})
            event = await asyncio.wait_for(media.receive(), 3)
            assert event["type"] == "flush" and event["generation"] == 2
            assert await asyncio.to_thread(robot.stopped.wait, 3)
            await terminal(wire)
            await created(wire, "barrier")
            await wire.emit({"type": "response.output_audio_transcript.done", "response_id": "barrier", "transcript": "done"})
            assert (await asyncio.wait_for(media.receive(), 3))["text"] == "done"
            assert robot.calls == []
    asyncio.run(scenario())


def test_http_stop_is_independent_of_blocked_action_and_disconnect_quarantines_worker():
    async def scenario():
        async with rig(gateway=True) as (robot, _runtime, domain, wire, _, _, gate, origin):
            robot.block = threading.Event()
            async with aiohttp.ClientSession(headers={"Authorization": "Bearer " + gate.token}) as client:
                binding = await (await client.post(origin + "/api/session", json={"robot_id": "fixture"})).json()
                ws = await client.ws_connect(origin + "/api/media?ticket=" + binding["ticket"])
                await created(wire); await call(wire); await terminal(wire)
                assert await asyncio.to_thread(robot.entered.wait, 3)
                response = await asyncio.wait_for(client.post(origin + "/api/stop"), 3)
                assert (await response.json())["latched"] and robot.stopped.is_set()
                assert not robot.block.is_set()
                await ws.close()
                await asyncio.wait_for(gate.session.done.wait(), 3)
                assert not (await domain.close())["ok"]
                with pytest.raises(ValueError, match="closed|previous"):
                    domain.claim("next")
                robot.block.set()
    asyncio.run(scenario())


def test_gateway_auth_cross_origin_media_replay_and_reconnect():
    async def scenario():
        async with rig(gateway=True) as (robot, runtime, _, _wire, _, _, gate, origin):  # noqa: SIM117 — fixture bindings create client credentials
            async with aiohttp.ClientSession() as client:
                assert (await client.post(origin + "/api/stop")).status == 403
                headers = {"Authorization": "Bearer " + gate.token}
                assert (await client.post(origin + "/api/stop", headers={**headers, "Origin": "https://evil.invalid"})).status == 403
                assert (await client.post(origin + "/api/session", headers=headers, json={"robot_id": "other"})).status == 400
                one = await (await client.post(origin + "/api/session", headers=headers, json={"robot_id": "fixture"})).json()
                ws = await client.ws_connect(origin + "/api/media?ticket=" + one["ticket"])
                with pytest.raises(aiohttp.WSServerHandshakeError):
                    await client.ws_connect(origin + "/api/media?ticket=" + one["ticket"])
                await ws.send_json({"type": "audio", "session_id": one["session_id"], "sequence": 1,
                                    "audio": base64.b64encode(pcm_fixture()).decode()})
                await asyncio.wait_for(gate.session.done.wait(), 3)
                assert runtime.stopped and robot.calls == []
                await ws.close()
                assert (await (await client.post(origin + "/api/reset", headers=headers)).json())["ok"]
                two = await (await client.post(origin + "/api/session", headers=headers, json={"robot_id": "fixture"})).json()
                assert one["session_id"] != two["session_id"]
                async with client.ws_connect(origin + "/api/media?ticket=" + two["ticket"]) as ws2:
                    await ws2.send_json({"type": "text", "session_id": one["session_id"], "text": "old"})
                    await asyncio.wait_for(gate.session.done.wait(), 3)
                    assert runtime.stopped and robot.calls == []
    asyncio.run(scenario())


def test_http_stop_during_provider_handshake_does_not_wait_for_provider_ack():
    async def scenario():
        async with rig(gateway=True) as (robot, runtime, _, wire, _, _, gate, origin):
            wire.ack.clear()
            async with aiohttp.ClientSession(headers={"Authorization": "Bearer " + gate.token}) as client:
                create_task = asyncio.create_task(client.post(origin + "/api/session", json={"robot_id": "fixture"}))
                await asyncio.wait_for(wire.updated.wait(), 3)
                stop = await asyncio.wait_for(client.post(origin + "/api/stop"), 3)
                assert (await stop.json())["latched"] and robot.stopped.is_set()
                assert not wire.ack.is_set() and not create_task.done()
                wire.ack.set()
                assert (await create_task).status == 200
                assert runtime.stopped  # handshake success never resets the latch
    asyncio.run(scenario())


def test_incompatible_provider_audio_format_is_rejected_before_capture():
    async def scenario():
        async with rig(gateway=True) as (robot, _, _, wire, _, _, gate, origin):
            wire.bad_format = True
            async with aiohttp.ClientSession(headers={"Authorization": "Bearer " + gate.token}) as client:
                response = await client.post(origin + "/api/session", json={"robot_id": "fixture"})
                assert response.status == 502
                assert gate.session.closed and robot.calls == [] and robot.stopped.is_set()
    asyncio.run(scenario())


def test_late_new_response_after_external_stop_reset_cannot_refresh_session_authority():
    async def scenario():
        async with rig() as (robot, runtime, _, wire, _, _, *_):
            assert runtime.stop()["ok"] and runtime.reset_stop()["ok"]
            # An old input's generation can arrive only after reset. Its new
            # provider response ID must still retain the connection's old token.
            await created(wire); await call(wire); await terminal(wire)
            result = await wire.next("conversation.item.create")
            assert "stale execution generation" in json.loads(result["item"]["output"])["error"]
            assert robot.calls == []
    asyncio.run(scenario())


def test_barge_in_while_response_creation_is_delayed_invalidates_old_input():
    async def scenario():
        async with rig() as (robot, runtime, _, wire, session, media, *_):
            await session.text("walk forward")
            await wire.next("response.create")
            await wire.emit({"type": "input_audio_buffer.speech_started"})
            await asyncio.wait_for(media.receive(), 3)
            assert runtime.stopped
            await created(wire); await call(wire); await terminal(wire)
            result = await wire.next("conversation.item.create")
            assert "stale execution generation" in json.loads(result["item"]["output"])["error"]
            assert robot.calls == []
    asyncio.run(scenario())


def test_stop_reaches_runtime_before_a_blocked_media_flush():
    async def scenario():
        async with rig() as (robot, runtime, _, _wire, session, media, *_):
            entered, release = asyncio.Event(), asyncio.Event()
            original_flush = media.flush
            async def blocked_flush(reason):
                entered.set()
                await release.wait()
                await original_flush(reason)
            media.flush = blocked_flush
            pending = asyncio.create_task(session.interrupt("operator_stop", send_cancel=False, force_stop=True))
            await asyncio.wait_for(entered.wait(), 3)
            assert runtime.stopped and robot.stopped.is_set() and not pending.done()
            release.set()
            assert (await asyncio.wait_for(pending, 3))["media_flush_ok"]
    asyncio.run(scenario())


def test_simultaneous_session_posts_reserve_ownership_before_awaiting_body(monkeypatch):
    async def scenario():
        async with rig(gateway=True) as (_robot, _runtime, _domain, wire, _, _, gate, origin):
            body_entered, body_release = asyncio.Event(), asyncio.Event()
            original = web.Request.json
            async def slow_json(request, *args, **kwargs):
                if request.path == "/api/session":
                    body_entered.set()
                    await body_release.wait()
                return await original(request, *args, **kwargs)
            monkeypatch.setattr(web.Request, "json", slow_json)
            async with aiohttp.ClientSession(headers={"Authorization": "Bearer " + gate.token}) as client:
                first = asyncio.create_task(client.post(origin + "/api/session", json={"robot_id": "fixture"}))
                await asyncio.wait_for(body_entered.wait(), 3)
                second = await asyncio.wait_for(client.post(origin + "/api/session", json={"robot_id": "fixture"}), 3)
                assert second.status == 409
                body_release.set()
                assert (await first).status == 200
                assert len(wire.sessions) == 1
                owned = gate.session
                await gate.close()
                assert owned.closed and owned.provider._ws is None
    asyncio.run(scenario())


def test_cli_builds_real_synthetic_profile_and_serves_tool_result(tmp_path):
    import os
    import sys
    from pathlib import Path

    async def scenario():
        async with rig(gateway=True) as (_robot, _runtime, _domain, wire, _, _, fixture_gate, _origin):
            repo = Path(__file__).resolve().parents[1]
            run_dir = tmp_path / "conversation-cli"
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "cascade.apps.conversation", "--provider-url",
                fixture_gate.provider_factory().config.url, "--port", "0", "--run-dir", str(run_dir),
                "--allow-tool", "sensing.read_sensor", cwd=repo,
                env={**os.environ, "PYTHONPATH": str(repo / "src")},
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                line = await asyncio.wait_for(process.stdout.readline(), 15)
                assert line.startswith(b"Open "), line
                url, token = line.decode().strip()[5:].split("/#")
                async with aiohttp.ClientSession(headers={"Authorization": "Bearer " + token}) as client:
                    status = await (await client.get(url + "/api/status")).json()
                    assert status["robot_id"] == "conversation_mock"
                    binding = await (await client.post(url + "/api/session", json={"robot_id": "conversation_mock"})).json()
                    async with client.ws_connect(url + "/api/media?ticket=" + binding["ticket"]) as ws:
                        await ws.send_json({"type": "text", "session_id": binding["session_id"], "text": "read the IMU"})
                        await wire.next("response.create")
                        await created(wire)
                        await call(wire, args={"sensor_id": "imu"})
                        await terminal(wire)
                        output = await wire.next("conversation.item.create")
                        result = json.loads(output["item"]["output"])
                        assert result["ok"]
                        assert result["observation"]["measurement_kind"] == "synthetic"
            finally:
                if process.returncode is None:
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 15)
                except TimeoutError:
                    process.kill()
                    await process.wait()
                    raise
            assert process.returncode == 0, (await process.stderr.read()).decode()
            closure = json.loads((run_dir / "closure.json").read_text())
            assert closure["conversation"]["ok"] and closure["runtime"]["ok"]
    asyncio.run(scenario())
