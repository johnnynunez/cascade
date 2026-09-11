"""Opt-in REAL OpenClaw/Cosmos chat regression; read-only CASCADE world_state.

Run on Spark with COSMOS_OPENCLAW_LIVE=1 CASCADE_OPENCLAW_PROFILE=cascade-demo
and the private OpenClaw binary on PATH. Requires the exclusive profile's
allowlist to be exactly cascade__world_state; this test never enables motion.
COSMOS_OPENCLAW_EVIDENCE selects a retained artifact directory.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import time
import uuid

import pytest

REPO = Path(__file__).resolve().parents[1]
MODEL = "custom-127-0-0-1-8082/cosmos3-edge"


@pytest.mark.parametrize("include_ok", [False, True], ids=["world-first", "ok-then-world"])
def test_real_openclaw_executes_world_state(include_ok):
    if os.environ.get("COSMOS_OPENCLAW_LIVE") != "1":
        pytest.skip("explicit live read-only OpenClaw/Cosmos opt-in required")
    assert os.environ.get("CASCADE_OPENCLAW_PROFILE") == "cascade-demo"
    config = json.loads((Path.home() / ".openclaw-cascade-demo/openclaw.json").read_text())
    assert config["gateway"]["port"] == 18790
    assert config["tools"]["allow"] == ["cascade__world_state"], "deny motion tools before this live check"
    assert config["models"]["providers"][MODEL.split("/")[0]]["baseUrl"] == "http://127.0.0.1:8082/v1"
    spec = importlib.util.spec_from_file_location("cosmos_chat_demo_proof", REPO / "scripts/demo_proof.py")
    assert spec and spec.loader
    proof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proof)
    session = "cosmos-chat-" + uuid.uuid4().hex
    evidence = Path(os.environ.get("COSMOS_OPENCLAW_EVIDENCE", str(REPO / "runs/cosmos-chat-final"))) / session
    evidence.mkdir(parents=True)
    state = REPO / "runs/.launch/profile-cascade-demo"
    from cascade.apps.process_owner import load_owner, records
    owner = load_owner(state, REPO, "cascade-demo")
    assert owner is not None
    baseline = {r.get("instance_id") for r in records(state)}
    started = time.time()
    if include_ok:
        brain = proof.agent_turn(session, MODEL, "Reply with exactly OK and nothing else. Do not call tools.", evidence / "01-brain.json", 90)
        assert " ".join(p.get("text", "") for p in brain.get("payloads", [])).strip() == "OK"
    world = proof.agent_turn(
        session, MODEL,
        "Call CASCADE world_state exactly once as the initial health check. Do not move the robot. "
        "Do not use system commands or other MCP servers." + ("" if include_ok else " Report the returned result."),
        evidence / "02-world.json", 90, "world_state",
    )
    meta = world["meta"]
    assert meta["toolSummary"] == {"calls": 1, "failures": 0, "tools": ["cascade__world_state"]}
    assert meta["agentMeta"]["terminalReceipt"]["successfulToolNames"] == ["cascade__world_state"]
    assert meta["agentMeta"]["terminalReceipt"]["rerouted"] is False
    process = proof._bound_world(state, owner, baseline, started)
    # Read back the actual host transcript: prose that looks like JSON is NOT a call.
    import sqlite3
    database = Path.home() / ".openclaw-cascade-demo/agents/main/agent/openclaw-agent.sqlite"
    with sqlite3.connect("file:" + str(database) + "?mode=ro", uri=True) as connection:
        events = [json.loads(row[0]) for row in connection.execute(
            "SELECT event_json FROM transcript_events WHERE session_id=? ORDER BY seq", (session,))]
    (evidence / "native-transcript.json").write_text(json.dumps(events, indent=2) + "\n")
    messages = [event["message"] for event in events if event.get("type") == "message"]
    calls = [part for message in messages if message.get("role") == "assistant"
             for part in message.get("content", []) if isinstance(part, dict) and part.get("type") == "toolCall"]
    assert len(calls) == 1 and calls[0]["name"] == "cascade__world_state"
    assert calls[0]["arguments"] == {}
    results = [message for message in messages if message.get("role") == "toolResult"]
    assert len(results) == 1 and results[0]["toolCallId"] == calls[0]["id"]
    assert results[0]["isError"] is False
    assert results[0]["details"] == {"mcpServer": "cascade", "mcpTool": "world_state"}
    tool_result = json.loads(results[0]["content"][0]["text"])
    assert tool_result["ok"] is True
    report = {"verified": True, "scope": "read-only host world_state; no motion", "model": MODEL,
              "session_id": session, "evidence_dir": str(evidence), "process": process,
              "native_call": calls[0], "tool_result": tool_result,
              "toolSummary": meta["toolSummary"], "usage": meta["agentMeta"].get("usage"),
              "params": config["agents"]["defaults"]["models"][MODEL].get("params", {})}
    (evidence / "verified.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
