"""A staged stop must reach its owner before transport can retain prior output."""
import asyncio
import base64
import json
import threading
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.conversation import domain as domain_module
from cascade.conversation.domain import ConversationDomain
from cascade.conversation.media import QueueMediaIO
from cascade.conversation.session import ConversationSession
from cascade.robotics.contracts import ResourceDescriptor, ToolDescriptor
from cascade.robotics.runtime import RobotRuntime


class Owner:
    domain_id = "body"
    resources = (ResourceDescriptor("body/base", "base", "fixture", synthetic=True,
        admission="software_only", controller_id="fixture/controller", writer_id="body"),)

    def __init__(self):
        schema = {"type": "object", "properties": {}, "additionalProperties": False}
        self.tool_descriptors = (
            ToolDescriptor("body.walk_velocity", "synthetic motion", schema, "body", "walk_velocity",
                           effect="motion", writes=("body/base",)),
            ToolDescriptor("body.read", "synthetic read", schema, "body", "read"))
        self.calls = []
        self.stop_calls = 0
        self.stopped = threading.Event()
        self.entered, self.release = threading.Event(), threading.Event()
        self.release.set()

    def execute(self, name, args):
        self.calls.append((name, args))
        self.entered.set()
        assert self.release.wait(5), "test did not release its own operation"
        return {"ok": True, "synthetic": True, "postcondition": {"status": "unverified"}}

    def stop(self):
        self.stop_calls += 1
        self.stopped.set()
        return {"ok": True}

    def reset_stop(self):
        return {"ok": True}

    def close(self):
        return {"ok": True}


class HeldOutputProvider:
    def __init__(self):
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.sent = []

    async def send(self, event):
        if event.get("item", {}).get("type") == "function_call_output":
            self.entered.set()
            await self.release.wait()
        self.sent.append(event)

    async def close(self):
        self.release.set()

    def outputs(self):
        return [event["item"] for event in self.sent
                if event.get("item", {}).get("type") == "function_call_output"]


@asynccontextmanager
async def rig():
    owner = Owner()
    runtime = RobotRuntime({"body": owner})
    domain = ConversationDomain(runtime, robot_id="fixture", allow_motion=True,
        allow_tools=("body.walk_velocity", "body.read", "emergency_stop"))
    provider = HeldOutputProvider()
    session = ConversationSession(domain, provider, QueueMediaIO())
    domain.claim(session.session_id)
    session.ready = True
    try:
        yield owner, runtime, domain, provider, session
    finally:
        owner.release.set()
        provider.release.set()
        await session.close()
        await domain.close()
        runtime.close()


async def stage(session, names):
    await session.text("operator input")
    origin = session._pending.origin
    metadata = session.provider.sent[-1]["response"]["metadata"]
    await session.handle({"type": "response.created", "response": {"id": "response", "metadata": metadata}})
    aliases = {name: alias for alias, name in session.domain.aliases.items()}
    for index, name in enumerate(names):
        await session.handle({"type": "response.function_call_arguments.done", "response_id": "response",
            "call_id": f"call-{index}", "name": aliases[name], "output_index": index, "arguments": "{}"})
    return origin


async def complete(session, count):
    await session.handle({"type": "response.done", "response": {"id": "response", "status": "completed",
        "output": [{"type": "function_call", "call_id": f"call-{i}"} for i in range(count)]}})


def test_adjacent_staged_stop_delivered_before_blocked_prior_output():
    async def scenario():
        async with rig() as (owner, runtime, domain, provider, session):
            admitted = []
            dispatch = domain.dispatch
            async def observed(intent):
                admitted.append(intent)
                return await dispatch(intent)
            domain.dispatch = observed
            origin = await stage(session, ["body.walk_velocity", "emergency_stop"])
            assert not owner.calls and not owner.stopped.is_set()  # No execution from partial calls.
            await complete(session, 2)
            await asyncio.wait_for(provider.entered.wait(), 3)
            # Causal assertion: transport has entered its hold and cannot finish;
            # no scheduler timing threshold is used to infer stop delivery.
            assert owner.stopped.is_set()
            assert owner.stop_calls == 1 and runtime.stopped
            assert owner.calls == [("walk_velocity", {})]
            assert [intent.request_id for intent in admitted] == ["call-0", "call-1"]
            assert all(intent.runtime_generation == origin.runtime_generation and
                       intent.deadline_monotonic_s == origin.deadline for intent in admitted)
            assert not provider.outputs()
            provider.release.set()
            await session._action
            outputs = provider.outputs()
            assert [row["call_id"] for row in outputs] == ["call-0", "call-1"]
            assert json.loads(outputs[0]["output"])["synthetic"] is True
            assert json.loads(outputs[1]["output"])["physical_stop_verified"] is False
            assert session._pending.origin is origin  # Continuation does not renew authority.
            assert provider.sent[-1]["type"] == "response.create"
    asyncio.run(scenario())


def test_stop_waits_for_its_preceding_operation_to_return():
    async def scenario():
        async with rig() as (owner, _, _, provider, session):
            owner.release.clear()
            await stage(session, ["body.walk_velocity", "emergency_stop"])
            await complete(session, 2)
            assert await asyncio.to_thread(owner.entered.wait, 3)
            assert owner.stop_calls == 0 and not provider.entered.is_set()
            owner.release.set()
            await asyncio.wait_for(provider.entered.wait(), 3)
            assert owner.stop_calls == 1
            provider.release.set()
            await session._action
    asyncio.run(scenario())


@pytest.mark.parametrize("operator_interrupt", [False, True])
def test_staged_stop_allows_correlated_speech_without_renewing_motion_authority(operator_interrupt):
    """Protocol-only PCM fixture; this is no physical result or model inference."""
    async def scenario():
        async with rig() as (owner, runtime, _, provider, session):
            origin = await stage(session, ["body.walk_velocity", "emergency_stop"])
            await complete(session, 2)
            await asyncio.wait_for(provider.entered.wait(), 3)
            provider.release.set()
            await session._action
            assert runtime.stopped and runtime.cancellation_token != origin.runtime_generation
            assert session._pending.origin is origin
            metadata = provider.sent[-1]["response"]["metadata"]
            await session.handle({"type": "response.created", "response": {
                "id": "narration", "metadata": metadata}})
            context = session.responses["narration"]
            assert context.origin is origin  # Deadline and generation stay original.
            if operator_interrupt:
                await session.interrupt("operator_stop", force_stop=True)
            pcm = b"\x01\x00" * 48
            await session.handle({"type": "response.output_audio.delta", "response_id": "narration",
                                  "delta": base64.b64encode(pcm).decode()})
            queued = []
            while not session.media.outgoing.empty():
                queued.append(await session.media.receive())
            audio = [event for event in queued if event["type"] == "audio"]
            assert bool(audio) is not operator_interrupt
            if audio:
                assert base64.b64decode(audio[0]["audio"]) == pcm
            alias = next(key for key, value in session.domain.aliases.items() if value == "body.walk_velocity")
            await session.handle({"type": "response.function_call_arguments.done", "response_id": "narration",
                "call_id": "forbidden-second-motion", "name": alias, "output_index": 0, "arguments": "{}"})
            await session.handle({"type": "response.done", "response": {
                "id": "narration", "status": "completed",
                "output": [{"type": "function_call", "call_id": "forbidden-second-motion"}]}})
            if not operator_interrupt:
                await session._action
                rejection = json.loads(provider.outputs()[-1]["output"])
                assert not rejection["ok"] and "generation" in rejection["error"]
            else:
                assert session.authority_revoked and not context.valid
            assert owner.calls == [("walk_velocity", {})]
            assert runtime.stopped
    asyncio.run(scenario())


@pytest.mark.parametrize("rejection", ["expired", "catalog_changed", "generation_changed"])
def test_rejected_adjacent_stop_keeps_domain_admission(rejection, monkeypatch):
    async def scenario():
        async with rig() as (owner, runtime, domain, provider, session):
            origin = await stage(session, ["body.walk_velocity", "emergency_stop"])
            dispatch = domain.dispatch
            async def after_motion(intent):
                result = await dispatch(intent)
                if intent.tool == "body.walk_velocity":
                    if rejection == "expired":
                        monkeypatch.setattr(domain_module, "time", SimpleNamespace(monotonic=lambda: origin.deadline + 1))
                    elif rejection == "catalog_changed":
                        runtime.tool_descriptors["emergency_stop"] = replace(
                            runtime.tool_descriptors["emergency_stop"], description="changed catalog")
                    else:
                        runtime.reset_stop()  # An explicit independent owner changes generation.
                return result
            domain.dispatch = after_motion
            await complete(session, 2)
            await asyncio.wait_for(provider.entered.wait(), 3)
            assert owner.stop_calls == 0
            provider.release.set()
            await session._action
            outputs = provider.outputs()
            assert [row["call_id"] for row in outputs] == ["call-0", "call-1"]
            result = json.loads(outputs[1]["output"])
            assert result["ok"] is False
            expected = {"expired": "expired", "catalog_changed": "catalog changed", "generation_changed": "generation"}
            assert expected[rejection] in result["error"]
            assert owner.stop_calls == 0
    asyncio.run(scenario())


def test_context_cancelled_after_motion_does_not_dispatch_staged_stop():
    async def scenario():
        async with rig() as (owner, runtime, domain, provider, session):
            await stage(session, ["body.walk_velocity", "emergency_stop"])
            admitted = []
            dispatch = domain.dispatch
            async def cancelled(intent):
                admitted.append(intent.tool)
                result = await dispatch(intent)
                await session.interrupt("operator_stop", force_stop=True)
                return result
            domain.dispatch = cancelled
            await complete(session, 2)
            await asyncio.wait_for(session._action, 3)
            assert runtime.stopped and owner.stop_calls == 1  # Operator stop, no second tool delivery.
            assert admitted == ["body.walk_velocity"]
            assert not provider.outputs() and session._pending is None
    asyncio.run(scenario())


def test_no_stop_retains_ordinary_dispatch_then_output_order():
    async def scenario():
        async with rig() as (owner, _, _, provider, session):
            await stage(session, ["body.read", "body.walk_velocity"])
            await complete(session, 2)
            await asyncio.wait_for(provider.entered.wait(), 3)
            assert owner.calls == [("read", {})] and owner.stop_calls == 0
            provider.release.set()
            await session._action
            assert owner.calls == [("read", {}), ("walk_velocity", {})]
            assert [row["call_id"] for row in provider.outputs()] == ["call-0", "call-1"]
    asyncio.run(scenario())


def test_nonadjacent_stop_is_not_moved_ahead_of_an_intervening_action():
    async def scenario():
        async with rig() as (owner, _, _, provider, session):
            await stage(session, ["body.read", "body.walk_velocity", "emergency_stop"])
            await complete(session, 3)
            await asyncio.wait_for(provider.entered.wait(), 3)
            assert owner.calls == [("read", {})] and owner.stop_calls == 0
            provider.release.set()
            await session._action
            assert owner.calls == [("read", {}), ("walk_velocity", {})]
            assert owner.stop_calls == 1
            assert [row["call_id"] for row in provider.outputs()] == ["call-0", "call-1", "call-2"]
    asyncio.run(scenario())
