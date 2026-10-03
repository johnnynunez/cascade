"""Input-authority regressions: real wire plus deterministic native-event replay."""
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_conversation_protocol import call, created, rig, terminal

from cascade.conversation import session as session_module

FIXTURE = Path(__file__).parent / "fixtures/conversation_input_origin_20261002.json"


async def media_event(media, kind):
    while True:
        event = await asyncio.wait_for(media.receive(), 3)
        if event["type"] == kind:
            return event


async def direct_tool(session, rid, cid="c1"):
    await session.handle({"type": "response.function_call_arguments.done", "response_id": rid,
        "call_id": cid, "name": "robot_tool_0", "output_index": 0, "arguments": "{}"})
    await session.handle({"type": "response.done", "response": {"id": rid, "status": "completed",
        "output": [{"type": "function_call", "call_id": cid}]}})


@pytest.mark.parametrize("case", json.loads(FIXTURE.read_text())["cases"], ids=["native-fp16", "native-fp32"])
def test_native_delayed_speech_stop_does_not_refresh_input_deadline(monkeypatch, case):
    async def scenario():
        async with rig(allow_motion=False, intent_timeout_s=10) as (robot, _, _, _, session, media, *_):
            events = case["events"]
            # Replay exact event payloads/spacing in a session-local clock only.
            # End of replay is the real local present, so the real domain/runtime
            # admission clock rejects the injected tool after the original 10 s.
            clock = [time.monotonic() - (events[-1]["observed_monotonic"] - events[0]["observed_monotonic"])]
            start = clock[0]
            monkeypatch.setattr(session_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
            for event in events:
                clock[0] = start + event["observed_monotonic"] - events[0]["observed_monotonic"]
                await session.handle(event)
            await direct_tool(session, events[-1]["response"]["id"])
            result = (await media_event(media, "tool_result"))["result"]
            assert not result["ok"] and "expired" in result["error"]
            assert robot.calls == []
    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["stop_robot", "speech_only"])
def test_two_speech_onsets_then_old_stopped_created_tool_never_rebind(mode):
    async def scenario():
        async with rig(allow_motion=False, barge_in=mode) as (robot, runtime, _, wire, session, media, *_):
            generation = runtime.cancellation_token
            for item in ("old-input", "new-input"):
                await wire.emit({"type": "input_audio_buffer.speech_started", "item_id": item})
                await media_event(media, "flush")
                if item == "old-input":
                    assert not runtime.stopped  # First idle speech remains benign.
            revoked = await media_event(media, "authority_revoked")
            assert revoked["reconnect_required"] and session.authority_revoked
            assert runtime.stopped == (mode == "stop_robot")
            if mode == "speech_only":
                assert runtime.cancellation_token == generation and not robot.stopped.is_set()
            await wire.emit({"type": "input_audio_buffer.speech_stopped", "item_id": "old-input"})
            await created(wire, automatic=True)
            await call(wire, args={}); await terminal(wire)
            # Even an apparently new untagged response is ambiguous after barge-in.
            await wire.emit({"type": "input_audio_buffer.speech_stopped", "item_id": "new-input"})
            await created(wire, "new-response", automatic=True)
            await call(wire, rid="new-response", cid="c2", args={})
            await terminal(wire, rid="new-response", cid="c2")
            await wire.emit({"type": "conversation.item.input_audio_transcription.completed", "transcript": "barrier"})
            await media_event(media, "transcript")
            assert robot.calls == [] and not session.closed
            with pytest.raises(ValueError, match="reconnect"):
                await session.text("try another input")
    asyncio.run(scenario())


def test_first_vad_input_and_correlated_tool_continuation_keep_same_origin():
    async def scenario():
        async with rig(allow_motion=False) as (robot, runtime, _, wire, session, media, *_):
            await wire.emit({"type": "input_audio_buffer.speech_started", "item_id": "u1"})
            await media_event(media, "flush")
            origin = session._pending.origin
            assert not runtime.stopped
            await wire.emit({"type": "input_audio_buffer.speech_stopped", "item_id": "u1"})
            await created(wire, automatic=True); await call(wire, args={}); await terminal(wire)
            await media_event(media, "tool_result")
            followup = await wire.next("response.create")
            await created(wire, "r2", request=followup)
            await call(wire, rid="r2", cid="c2", args={}); await terminal(wire, rid="r2", cid="c2")
            await media_event(media, "tool_result")
            assert session.responses["r1"].origin is origin
            assert session.responses["r2"].origin is origin
            assert robot.calls == [("read", {}), ("read", {})]
            assert set(followup["response"]) == {"metadata"}  # No nonce in robot arguments.
    asyncio.run(scenario())


def test_tool_followup_never_gets_a_new_deadline(monkeypatch):
    async def scenario():
        async with rig(allow_motion=False) as (robot, _, _, wire, session, media, *_):
            # Freeze only the input-origin clock at the real present for admission.
            clock = [time.monotonic()]
            monkeypatch.setattr(session_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
            await created(wire); await call(wire, args={}); await terminal(wire)
            await media_event(media, "tool_result")
            request = await wire.next("response.create")
            await session._action
            original_deadline = session.responses["r1"].origin.deadline
            # Deterministic passage of local time for response processing: a
            # continuation's deadline must equal the input's, regardless of delay.
            clock[0] += 100
            await created(wire, "r2", request=request)
            await wire.emit({"type": "response.output_audio_transcript.done", "response_id": "r2", "transcript": "barrier"})
            await media_event(media, "transcript")
            assert session.responses["r2"].origin.deadline == original_deadline
            assert robot.calls == [("read", {})]
    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["spontaneous", "missing_nonce", "wrong_nonce", "replayed_nonce",
                                  "stop_before_start", "duplicate_start", "wrong_stop", "duplicate_stop"])
def test_unbound_or_replayed_provider_origin_closes_without_action(fault):
    async def scenario():
        async with rig(allow_motion=False) as (robot, _, _, wire, session, media, *_):
            if fault in {"missing_nonce", "wrong_nonce", "replayed_nonce"}:
                await session.text("read state")
                request = await wire.next("response.create")
                metadata = request["response"]["metadata"].copy()
                if fault == "replayed_nonce":
                    await created(wire, request=request)
                    await call(wire, args={}); await terminal(wire)
                    await media_event(media, "tool_result")
                    await wire.next("response.create")
                elif fault == "missing_nonce":
                    metadata = {}
                else:
                    metadata["cascade_request_id"] = "a-different-request"
                await wire.emit({"type": "response.created", "response": {"id": "bad", "metadata": metadata}})
            elif fault == "spontaneous":
                await created(wire, automatic=True)
            else:
                if fault != "stop_before_start":
                    await wire.emit({"type": "input_audio_buffer.speech_started", "item_id": "u1"})
                    await media_event(media, "flush")
                if fault == "duplicate_start":
                    await wire.emit({"type": "input_audio_buffer.speech_started", "item_id": "u1"})
                else:
                    await wire.emit({"type": "input_audio_buffer.speech_stopped", "item_id": "wrong" if fault == "wrong_stop" else "u1"})
                    if fault == "duplicate_stop":
                        await wire.emit({"type": "input_audio_buffer.speech_stopped", "item_id": "u1"})
            await asyncio.wait_for(session.done.wait(), 3)
            assert robot.calls == ([("read", {})] if fault == "replayed_nonce" else [])
    asyncio.run(scenario())


@pytest.mark.parametrize("interrupt", ["stop", "disconnect"])
def test_speech_start_cannot_reinstate_pending_authority_after_flush_race(interrupt):
    async def scenario():
        async with rig(allow_motion=False) as (robot, runtime, _, _wire, session, media, *_):
            entered, release = asyncio.Event(), asyncio.Event()
            original = media.flush
            async def held_flush(reason):
                if reason == "barge_in":
                    entered.set()
                    await release.wait()
                await original(reason)
            media.flush = held_flush
            pending = asyncio.create_task(session.handle({"type": "input_audio_buffer.speech_started", "item_id": "u1"}))
            await asyncio.wait_for(entered.wait(), 3)
            assert session._pending.origin.item_id == "u1"
            if interrupt == "stop":
                await session.interrupt("operator_stop", force_stop=True)
            else:
                await session.close("operator_disconnect")
            release.set()
            await pending
            assert runtime.stopped and robot.calls == []
            if not session.closed:
                assert session._pending is None and session.authority_revoked
    asyncio.run(scenario())


def test_text_request_held_during_cancel_cannot_send_a_fresh_response_request():
    async def scenario():
        async with rig(allow_motion=False) as (robot, runtime, _, _wire, session, _media, *_):
            entered, release = asyncio.Event(), asyncio.Event()
            original, sent = session.provider.send, []
            async def hold_input(event):
                sent.append(event["type"])
                if event["type"] == "conversation.item.create":
                    entered.set()
                    await release.wait()
                await original(event)
            session.provider.send = hold_input
            pending = asyncio.create_task(session.text("read state"))
            await asyncio.wait_for(entered.wait(), 3)
            await session.interrupt("operator_stop", force_stop=True)
            release.set()
            await pending
            assert "response.create" not in sent
            assert runtime.stopped and robot.calls == []
    asyncio.run(scenario())


def test_actual_pinned_hf_serialized_metadata_response_is_admitted(monkeypatch):
    fixture = json.loads((FIXTURE.parent / "conversation_hf_metadata_20261002.json").read_text())
    async def scenario():
        async with rig(allow_motion=False) as (robot, _, _, wire, session, media, *_):
            request = json.loads(fixture["request_json"])
            nonce = request["response"]["metadata"]["cascade_request_id"]
            # Match the correlation ID emitted by the isolated actual HF handler
            # probe; neither provider bytes nor tool arguments are rewritten.
            monkeypatch.setattr(session_module, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex=nonce)))
            await session.text("read state")
            assert await wire.next("response.create") == request
            await wire.ws.send_str(fixture["response_json"])
            response = json.loads(fixture["response_json"])["response"]
            await call(wire, rid=response["id"], args={})
            await terminal(wire, rid=response["id"])
            assert (await media_event(media, "tool_result"))["result"]["ok"]
            assert robot.calls == [("read", {})]
    asyncio.run(scenario())
