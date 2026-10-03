"""A declared stop tool keeps the existing independent runtime stop semantics."""
import asyncio
import concurrent.futures
import json
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import ClassVar

import pytest

from cascade.conversation.domain import ConversationDomain, ToolIntent
from cascade.robotics.contracts import ResourceDescriptor, ToolDescriptor
from cascade.robotics.runtime import RobotRuntime


class Owner:
    domain_id = "body"
    resources = (ResourceDescriptor("body/base", "base", "fixture", synthetic=True,
        admission="software_only", controller_id="fixture/controller", writer_id="body"),)
    schema: ClassVar[dict] = {"type": "object", "properties": {}, "additionalProperties": False}
    tool_descriptors = (
        ToolDescriptor("body.walk_velocity", "fixture motion", schema, "body", "walk_velocity",
                       effect="motion", writes=("body/base",)),
        ToolDescriptor("body.stop_navigation", "priority stop", schema, "body", "stop_navigation",
                       effect="stop"),
    )

    def __init__(self):
        self.entered, self.release, self.stopped = (threading.Event() for _ in range(3))
        self.calls = []
        self.stop_calls = 0

    def execute(self, name, args):
        self.calls.append((name, args))
        self.entered.set()
        assert self.release.wait(5), "test did not release action"
        return {"ok": True, "synthetic": True}

    def stop(self):
        self.stop_calls += 1
        self.stopped.set()
        return {"ok": True}

    def reset_stop(self):
        return {"ok": True}

    def close(self):
        return {"ok": True}


def setup():
    owner = Owner()
    runtime = RobotRuntime({"body": owner})
    domain = ConversationDomain(runtime, robot_id="fixture", allow_motion=True,
        allow_tools=("body.walk_velocity", "body.stop_navigation"))
    domain.claim("session")
    return owner, runtime, domain


def request(*, name="body.stop_navigation", call="stop", **changes):
    value = ToolIntent("session", "fixture", "response", call, 0,
                       time.monotonic() + 60, name, {})
    return replace(value, **changes)


def test_declared_stop_preempts_blocked_motion_without_replacing_its_worker():
    async def scenario():
        owner, runtime, domain = setup()
        task = asyncio.create_task(domain.dispatch(request(name="body.walk_velocity", call="motion")))
        try:
            assert await asyncio.to_thread(owner.entered.wait, 3)
            motion_future, motion_thread = domain._future, domain._thread
            another = await domain.dispatch(request(name="body.walk_velocity", call="another"))
            assert not another["ok"] and "in flight" in another["error"]
            result = await asyncio.wait_for(domain.dispatch(request()), 3)
            assert result["ok"] and result["latched"]
            assert result["physical_stop_verified"] is False
            assert runtime.stopped and owner.stopped.is_set()
            assert domain._future is motion_future and domain._thread is motion_thread
            assert domain.action_pending and not task.done()
            assert len(owner.calls) == 1 and owner.stop_calls == 1
            replay = await domain.dispatch(request())
            assert not replay["ok"] and "duplicate" in replay["error"]
            assert owner.stop_calls == 1
        finally:
            owner.release.set()
            await asyncio.wait_for(task, 3)
            await domain.close()
            runtime.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["session", "robot", "expired", "generation", "allowlist", "catalog", "schema"])
def test_stop_tool_keeps_admission_checks(fault):
    async def scenario():
        owner, runtime, domain = setup()
        value = request()
        if fault == "session":
            value = replace(value, session_id="old")
        elif fault == "robot":
            value = replace(value, robot_id="another")
        elif fault == "expired":
            value = replace(value, deadline_monotonic_s=time.monotonic() - 1)
        elif fault == "generation":
            value = replace(value, runtime_generation=3)
        elif fault == "allowlist":
            domain.tools.pop(value.tool)
        elif fault == "catalog":
            runtime.tool_descriptors[value.tool] = replace(domain.tools[value.tool], description="changed")
        elif fault == "schema":
            value = replace(value, arguments={"unexpected": True})
        try:
            try:
                result = await domain.dispatch(value)
            except ValueError:
                assert fault == "schema"
            else:
                assert not result["ok"]
            assert not runtime.stopped and owner.stop_calls == 0 and owner.calls == []
            assert domain._future is None
        finally:
            await domain.close()
            runtime.close()
    asyncio.run(scenario())


def test_declared_stop_uses_coalesced_stop_path():
    async def scenario():
        _owner, runtime, domain = setup()
        called = []
        original = domain.stop
        async def observed(**kwargs):
            called.append(kwargs)
            return await original(**kwargs)
        domain.stop = observed
        try:
            result = await domain.dispatch(request())
            assert called and result["ok"]
            assert domain._stop_task is not None
            assert domain._future is None and domain._thread is None
        finally:
            await domain.close()
            runtime.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["expired", "released", "catalog", "generation"])
def test_stop_rechecks_authority_at_final_local_admission(monkeypatch, fault):
    async def scenario():
        owner, runtime, domain = setup()
        entered, release = asyncio.Event(), asyncio.Event()
        original = domain.stop
        async def held(**kwargs):
            entered.set()
            await release.wait()
            return await original(**kwargs)
        domain.stop = held
        pending = asyncio.create_task(domain.dispatch(request()))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if fault == "expired":
                import cascade.conversation.domain as module
                # Change only this module binding; all real runtime/test clocks
                # retain stdlib time. Expiry occurs after initial validation.
                monotonic = time.monotonic
                monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: monotonic() + 61))
            elif fault == "released":
                domain.release("session")
            elif fault == "catalog":
                domain.tools.pop("body.stop_navigation")
            else:
                assert runtime.stop()["ok"]
                assert runtime.reset_stop()["ok"]
            before = owner.stop_calls
            release.set()
            result = await asyncio.wait_for(pending, 3)
            assert not result["ok"] and owner.stop_calls == before
            assert domain._future is None and domain._stop_task is None
            # Independent operator stop is still available, without an intent.
            assert (await original())["latched"] and owner.stop_calls == before + 1
        finally:
            release.set()
            await asyncio.wait_for(pending, 3)
            await domain.close()
            runtime.close()
    asyncio.run(scenario())


def test_operator_stop_joins_admitted_tool_stop_without_inheriting_authority_checks():
    async def scenario():
        owner, runtime, domain = setup()
        entered, release = threading.Event(), threading.Event()
        original = runtime.stop
        def held():
            entered.set()
            assert release.wait(5), "test did not release stop delivery"
            return original()
        runtime.stop = held
        tool = asyncio.create_task(domain.dispatch(request()))
        operator = None
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            admitted_task = domain._stop_task
            domain.release("session")
            operator = asyncio.create_task(domain.stop())
            await asyncio.sleep(0)
            assert domain._stop_task is admitted_task and not operator.done()
            release.set()
            first, second = await asyncio.wait_for(asyncio.gather(tool, operator), 3)
            assert first == second and first["physical_stop_verified"] is False
            assert owner.stop_calls == 1 and runtime.stopped
        finally:
            release.set()
            await asyncio.wait_for(tool, 3)
            if operator:
                await asyncio.wait_for(operator, 3)
            await domain.close()
            runtime.close()
    asyncio.run(scenario())


def test_admitted_stop_tool_retains_normal_runtime_trace(tmp_path):
    from cascade.agent.trace import TraceLogger
    async def scenario():
        owner, runtime, domain = setup()
        runtime.trace = TraceLogger(tmp_path)
        try:
            result = await domain.dispatch(request())
            rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
            assert len(rows) == 1
            assert rows[0]["skill"] == "body.stop_navigation" and rows[0]["args"] == {}
            assert rows[0]["result"] == result and result["physical_stop_verified"] is False
            assert rows[0]["context"]["robot_mode"] == "composed"
            assert owner.stop_calls == 1
        finally:
            await domain.close()
            runtime.close()
    asyncio.run(scenario())


def test_blocked_trace_never_delays_physical_stop_delivery():
    async def scenario():
        owner, runtime, domain = setup()
        entered, release = threading.Event(), threading.Event()
        class HeldTrace:
            def record(self, *args, **kwargs):
                assert owner.stopped.is_set() and runtime.stopped
                entered.set()
                assert release.wait(5), "test did not release trace"
        runtime.trace = HeldTrace()
        pending = asyncio.create_task(domain.dispatch(request()))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            assert runtime.stopped and owner.stop_calls == 1
            assert not pending.done() and domain._future is None
            # Operator requests do not inherit a tool's blocked logger.
            assert (await asyncio.wait_for(domain.stop(), 3))["latched"]
            assert owner.stop_calls == 2 and not pending.done()
            release.set()
            assert (await asyncio.wait_for(pending, 3))["physical_stop_verified"] is False
        finally:
            release.set()
            await asyncio.wait_for(pending, 3)
            await domain.close()
            runtime.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("first", ["operator", "tool"])
def test_every_tool_joining_coalesced_stop_retains_its_trace(tmp_path, first):
    from cascade.agent.trace import TraceLogger
    async def scenario():
        owner, runtime, domain = setup()
        runtime.trace = TraceLogger(tmp_path)
        entered, release = threading.Event(), threading.Event()
        original = runtime.stop
        def held():
            entered.set()
            assert release.wait(5), "test did not release stop delivery"
            return original()
        runtime.stop = held
        pending = [asyncio.create_task(domain.stop() if first == "operator" else
                                      domain.dispatch(request(call="first")))]
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            admitted_task = domain._stop_task
            pending.append(asyncio.create_task(domain.dispatch(request(call="second"))))
            await asyncio.sleep(0)
            assert domain._stop_task is admitted_task
            release.set()
            results = await asyncio.wait_for(asyncio.gather(*pending), 3)
            assert all(r["ok"] and r["physical_stop_verified"] is False for r in results)
            assert owner.stop_calls == 1
            rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
            assert len(rows) == (1 if first == "operator" else 2)
            assert all(r["skill"] == "body.stop_navigation" and r["args"] == {} and
                       r["result"] == results[0] for r in rows)
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(*pending), 3)
            await domain.close()
            runtime.close()
    asyncio.run(scenario())


def test_cancelled_tool_consumer_does_not_drop_its_stop_trace(tmp_path):
    from cascade.agent.trace import TraceLogger
    async def scenario():
        owner, runtime, domain = setup()
        real_trace = TraceLogger(tmp_path)
        recorded = threading.Event()
        class Trace:
            def record(self, *args, **kwargs):
                real_trace.record(*args, **kwargs)
                recorded.set()
        runtime.trace = Trace()
        entered, release = threading.Event(), threading.Event()
        original = runtime.stop
        def held():
            entered.set()
            assert release.wait(5), "test did not release stop delivery"
            return original()
        runtime.stop = held
        pending = asyncio.create_task(domain.dispatch(request()))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            release.set()
            assert await asyncio.to_thread(recorded.wait, 3), "admitted stop lost its trace with consumer cancellation"
            rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
            assert len(rows) == 1 and rows[0]["skill"] == "body.stop_navigation"
            assert rows[0]["result"]["physical_stop_verified"] is False
            assert owner.stop_calls == 1 and runtime.stopped
        finally:
            release.set()
            await domain.close()
            runtime.close()
    asyncio.run(scenario())


def test_close_reports_pending_stop_record_and_refuses_an_overlapping_session():
    async def scenario():
        owner, runtime, domain = setup()
        entered, release = threading.Event(), threading.Event()
        class HeldTrace:
            def record(self, *args, **kwargs):
                entered.set()
                assert release.wait(5), "test did not release trace"
        runtime.trace = HeldTrace()
        pending = asyncio.create_task(domain.dispatch(request()))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            domain.release("session")
            with pytest.raises(ValueError, match="records are pending"):
                domain.claim("next_session")
            receipt = await asyncio.wait_for(domain.close(), 3)
            assert not receipt["ok"] and not receipt["action_pending"]
            assert receipt["stop_tool_records_pending"] == 1 and not receipt["stop_tool_record_errors"]
            assert owner.stopped.is_set()
            release.set()
            await asyncio.wait_for(asyncio.gather(*domain._stop_records), 3)
            assert (await domain.close())["ok"]
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(*domain._stop_records), 3)
            await domain.close()
            runtime.close()
    asyncio.run(scenario())


def test_stop_record_error_is_retained_without_undoing_physical_stop():
    async def scenario():
        owner, runtime, domain = setup()
        class FailedTrace:
            def record(self, *args, **kwargs):
                raise OSError("fixture log unavailable")
        runtime.trace = FailedTrace()
        try:
            with pytest.raises(OSError, match="fixture log unavailable"):
                await domain.dispatch(request())
            assert runtime.stopped and owner.stop_calls == 1
            receipt = await domain.close()
            assert not receipt["ok"] and receipt["stop_tool_records_pending"] == 0
            assert receipt["stop_tool_record_errors"] == ["OSError"]
        finally:
            runtime.close()
    asyncio.run(scenario())


def test_operator_stop_has_capacity_when_the_default_executor_is_full():
    async def scenario():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1))
        owner, runtime, domain = setup()
        entered, delivered = asyncio.Event(), asyncio.Event()
        release = threading.Event()
        class HeldTrace:
            def record(self, *args, **kwargs):
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(5), "test did not release trace"
        runtime.trace = HeldTrace()
        original = owner.stop
        def counted_stop():
            result = original()
            if owner.stop_calls >= 2:
                loop.call_soon_threadsafe(delivered.set)
            return result
        owner.stop = counted_stop
        tool = asyncio.create_task(domain.dispatch(request()))
        operator = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert owner.stop_calls == 1
            operator = asyncio.create_task(domain.stop())
            await asyncio.wait_for(delivered.wait(), 3)
            assert (await asyncio.wait_for(operator, 3))["latched"]
            assert not release.is_set() and not tool.done()
        finally:
            release.set()
            await asyncio.wait_for(tool, 3)
            if operator:
                await asyncio.wait_for(operator, 3)
            await domain.close()
            runtime.close()
    asyncio.run(scenario())


def test_coalesced_operator_result_cannot_mutate_pending_tool_record(tmp_path):
    from cascade.agent.trace import TraceLogger
    async def scenario():
        owner, runtime, domain = setup()
        real_trace = TraceLogger(tmp_path)
        logging, release_log = threading.Event(), threading.Event()
        class HeldTrace:
            def record(self, *args, **kwargs):
                logging.set()
                assert release_log.wait(5), "test did not release trace"
                real_trace.record(*args, **kwargs)
        runtime.trace = HeldTrace()
        entered, release_stop = threading.Event(), threading.Event()
        original = runtime.stop
        def held():
            entered.set()
            assert release_stop.wait(5), "test did not release stop"
            return original()
        runtime.stop = held
        operator = asyncio.create_task(domain.stop())
        tool = None
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            tool = asyncio.create_task(domain.dispatch(request()))
            await asyncio.sleep(0)
            release_stop.set()
            receipt = await asyncio.wait_for(operator, 3)
            assert await asyncio.to_thread(logging.wait, 3)
            receipt["latched"] = False
            receipt["domains"]["body"]["ok"] = False
            release_log.set()
            value = await asyncio.wait_for(tool, 3)
            row = json.loads((tmp_path / "trace.jsonl").read_text())
            assert value["latched"] and value["domains"]["body"]["ok"]
            assert row["result"] == value and owner.stop_calls == 1
        finally:
            release_stop.set(); release_log.set()
            await asyncio.wait_for(operator, 3)
            if tool:
                await asyncio.wait_for(tool, 3)
            await domain.close()
            runtime.close()
    asyncio.run(scenario())


def test_closed_domain_does_not_present_cached_ack_as_a_new_stop():
    async def scenario():
        owner, runtime, domain = setup()
        try:
            closed = await domain.close()
            assert closed["ok"] and closed["stop_worker_closed"]
            assert runtime.reset_stop()["ok"]  # another owner's explicit reset
            generation, count = runtime.cancellation_token, owner.stop_calls
            result = await domain.stop()
            assert not result["ok"] and "no new stop" in result["error"]
            assert not runtime.stopped and runtime.cancellation_token == generation
            assert owner.stop_calls == count
            repeated = await domain.close()
            assert repeated == closed and owner.stop_calls == count
            closed["stop"]["latched"] = False
            assert (await domain.close())["stop"]["latched"] is True
        finally:
            runtime.close()
    asyncio.run(scenario())


def test_close_cannot_retire_executor_while_a_newer_operator_stop_is_pending():
    async def scenario():
        loop = asyncio.get_running_loop()
        _owner, runtime, domain = setup()
        first_returned, continue_close, second_entered = (asyncio.Event() for _ in range(3))
        release_second = threading.Event()
        original_stop, original_runtime_stop = domain.stop, runtime.stop
        closing = None
        attempts = 0
        def held_delivery():
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                loop.call_soon_threadsafe(second_entered.set)
                assert release_second.wait(5), "test did not release second stop"
            return original_runtime_stop()
        runtime.stop = held_delivery
        async def hold_close_after_first_delivery(**kwargs):
            result = await original_stop(**kwargs)
            if asyncio.current_task() is closing:
                first_returned.set()
                await continue_close.wait()
            return result
        domain.stop = hold_close_after_first_delivery
        closing = asyncio.create_task(domain.close())
        operator = None
        try:
            await asyncio.wait_for(first_returned.wait(), 3)
            dedicated = domain._stop_executor
            operator = asyncio.create_task(domain.stop())
            await asyncio.wait_for(second_entered.wait(), 3)
            continue_close.set()
            receipt = await asyncio.wait_for(closing, 3)
            assert not receipt["ok"] and receipt["stop_delivery_pending"]
            assert not receipt["stop_worker_closed"] and domain._stop_executor is dedicated
            assert not operator.done() and not release_second.is_set()
            release_second.set()
            assert (await asyncio.wait_for(operator, 3))["latched"]
            assert (await domain.close())["stop_worker_closed"]
        finally:
            continue_close.set(); release_second.set()
            await asyncio.wait_for(closing, 3)
            if operator:
                await asyncio.wait_for(operator, 3)
            await domain.close()
            runtime.close()
    asyncio.run(scenario())
