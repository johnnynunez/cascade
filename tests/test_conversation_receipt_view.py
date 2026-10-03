"""Speech transport retains verdicts; full observations remain in ordinary traces."""
import asyncio
import copy
import hashlib
import json
import threading

import pytest

from cascade.agent.trace import TraceLogger
from cascade.conversation import session as session_module
from cascade.conversation.receipts import OUTPUT_BYTES, speech_tool_output
from test_conversation_stop_order import complete, rig, stage


def receipt(status="confirmed"):
    return {"ok": status == "confirmed", "execution_ok": True,
        "outcome": status, "receipt_id": "receipt-exact", "task_id": "task-exact",
        "requested_distance_m": .03, "measured_distance_m": .025,
        "stop_ack": {"latched": True, "physical_stop_verified": False, "generation": 3},
        "measured": {"before": {"step": 1}, "after": {"step": 50},
                     "samples": [{"step": i, "contacts": "x"*1000} for i in range(60)]},
        "postcondition": {"status": status, "reason": "independent observed outcome",
            "metrics": {"distance_m": .025}, "limits": {"error_m": .008},
            "admission": {"epoch": "same-epoch", "model_identity_sha256": "a"*64},
            "completion": {"step": 50}, "evidence": {"samples": "y"*60000}},
        "image_jpeg_b64": "z"*40000}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@pytest.mark.parametrize("status", ["confirmed", "refuted", "unverified"])
def test_large_receipt_preserves_verdict_metrics_bindings_and_exact_omission_digests(status):
    original = receipt(status)
    saved = copy.deepcopy(original)
    output = speech_tool_output(original, recorded=True)
    assert len(output.encode()) <= OUTPUT_BYTES
    view = json.loads(output)
    retained, transport = view["result"], view["speech_transport"]
    assert transport["full_result_sha256"] == digest(original)
    omissions = {row["path"]: row for row in transport["omitted"]}
    assert set(omissions) == {"/image_jpeg_b64", "/measured/samples", "/postcondition/evidence"}
    for path, descriptor in omissions.items():
        value = original
        for key in path.split("/")[1:]:
            value = value[key]
        assert descriptor["json_sha256"] == digest(value)
        assert descriptor["json_bytes"] == len(json.dumps(value, allow_nan=False).encode())
        assert descriptor["items"] == len(value)
    expected = copy.deepcopy(original)
    del expected["image_jpeg_b64"], expected["measured"]["samples"], expected["postcondition"]["evidence"]
    assert retained == expected and original == saved
    assert retained["postcondition"]["status"] == status
    assert retained["stop_ack"]["physical_stop_verified"] is False


@pytest.mark.parametrize("value", [{"ok": True}, {"ok": False, "error": "refuted"},
    {"ok": True, "postcondition": {"status": "unverified", "evidence": [1, 2]}}])
def test_small_receipt_serialization_is_byte_identical(value):
    assert speech_tool_output(value) == json.dumps(value, allow_nan=False)


def test_large_receipt_without_trace_keeps_explicit_failure():
    result = json.loads(speech_tool_output(receipt()))
    assert result["ok"] is False and "transport bound" in result["error"]


@pytest.mark.parametrize("key", ["error", "unexpected_evidence", "postcondition"])
def test_unknown_or_semantic_large_fields_are_not_silently_cut(key):
    value = receipt()
    value[key] = "λ"*40000
    result = json.loads(speech_tool_output(value, recorded=True))
    assert result["ok"] is False and "transport bound" in result["error"]


def test_invalid_numeric_evidence_is_not_hidden_by_compaction():
    value = receipt()
    value["postcondition"]["evidence"]["invalid"] = float("nan")
    with pytest.raises(ValueError):
        speech_tool_output(value, recorded=True)


def test_ordinary_runtime_keeps_full_trace_and_sends_bounded_same_verdict(tmp_path):
    async def scenario():
        async with rig() as (owner, runtime, domain, provider, session):
            original = receipt("refuted")
            owner.execute = lambda *_: copy.deepcopy(original)
            runtime.trace = TraceLogger(tmp_path)
            provider.release.set()
            origin = await stage(session, ["body.walk_velocity"])
            await complete(session, 1)
            await asyncio.wait_for(session._action, 3)
            [message] = provider.outputs()
            assert message["call_id"] == "call-0"
            view = json.loads(message["output"])
            assert view["result"]["postcondition"]["status"] == "refuted"
            [record] = [json.loads(line) for line in (tmp_path/"trace.jsonl").read_text().splitlines()]
            assert record["result"] == original
            assert view["speech_transport"]["full_result_sha256"] == digest(record["result"])
            assert runtime.unverified_actions()
            assert session._pending.origin is origin
    asyncio.run(scenario())


def test_adjacent_stop_runs_before_receipt_projection_and_does_not_rewrite_ack(tmp_path, monkeypatch):
    async def scenario():
        async with rig() as (owner, runtime, domain, provider, session):
            owner.execute = lambda *_: receipt("unverified")
            runtime.trace = TraceLogger(tmp_path)
            original = session_module.speech_tool_output
            def checked(result, **kwargs):
                assert owner.stopped.is_set()
                return original(result, **kwargs)
            monkeypatch.setattr(session_module, "speech_tool_output", checked)
            await stage(session, ["body.walk_velocity", "emergency_stop"])
            await complete(session, 2)
            await asyncio.wait_for(provider.entered.wait(), 3)
            assert runtime.stopped
            provider.release.set()
            await asyncio.wait_for(session._action, 3)
            outputs = provider.outputs()
            assert [row["call_id"] for row in outputs] == ["call-0", "call-1"]
            assert json.loads(outputs[0]["output"])["result"]["postcondition"]["status"] == "unverified"
            assert json.loads(outputs[1]["output"])["physical_stop_verified"] is False
    asyncio.run(scenario())


def test_operator_stop_passes_held_projection_and_cancelled_output_is_not_sent(tmp_path, monkeypatch):
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        async with rig() as (owner, runtime, domain, provider, session):
            owner.execute = lambda *_: receipt()
            runtime.trace = TraceLogger(tmp_path)
            original = session_module.speech_tool_output
            def held(result, **kwargs):
                entered.set()
                assert release.wait(5)
                return original(result, **kwargs)
            monkeypatch.setattr(session_module, "speech_tool_output", held)
            try:
                await stage(session, ["body.walk_velocity"])
                await complete(session, 1)
                assert await asyncio.to_thread(entered.wait, 3)
                await asyncio.wait_for(session.interrupt(force_stop=True), 3)
                assert owner.stopped.is_set() and runtime.stopped
                release.set()
                await asyncio.wait_for(session._action, 3)
                assert not provider.outputs() and not provider.entered.is_set()
                assert session.authority_revoked
            finally:
                release.set()
    asyncio.run(scenario())
