"""Opt-in replay of an OpenClaw HTTP body; generates, never executes tools.

Set COSMOS_TEST_BASE_URL and COSMOS_TEST_OPENCLAW_REQUEST to a captured JSON
request body (or an evidence object containing `request`). This is deliberately
not a fabricated compact prompt. The full workspace/context and tool list reach
the real server unchanged. Passing proves only the first native model call,
not MCP dispatch, a grounded final answer, or robot behavior.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

import pytest


def test_live_cosmos_openclaw_request_emits_native_world_state():
    base = os.environ.get("COSMOS_TEST_BASE_URL")
    source = os.environ.get("COSMOS_TEST_OPENCLAW_REQUEST")
    if not base or not source:
        pytest.skip("requires explicit local Cosmos endpoint and captured OpenClaw request")
    assert urlsplit(base).hostname in {"127.0.0.1", "localhost", "::1"}
    saved = json.loads(Path(source).read_text())
    payload = saved.get("request", saved)
    assert payload["model"] == "cosmos3-edge"
    assert payload["tool_choice"] == "auto", "a forced-call request is not this regression"
    assert any(t["function"]["name"] == "cascade__world_state" for t in payload["tools"])
    assert payload["messages"], "never replace the captured context with a smoke prompt"
    body = json.dumps(payload).encode()
    opener = build_opener(ProxyHandler({}))
    with opener.open(Request(base.rstrip("/") + "/chat/completions", data=body,
                             headers={"Content-Type": "application/json"}), timeout=180) as response:
        wire = response.read()
    if dest := os.environ.get("COSMOS_TEST_EVIDENCE_DIR"):
        path = Path(dest)
        path.mkdir(parents=True, exist_ok=True)
        (path / "openclaw-replay.json").write_text(json.dumps({
            "source": str(Path(source).resolve()),
            "request_sha256": hashlib.sha256(body).hexdigest(),
            "request": payload, "response_wire": wire.decode(),
            "tool_execution": False,
        }, indent=2))
    if payload.get("stream"):
        events = [json.loads(line[6:]) for line in wire.splitlines()
                  if line.startswith(b"data: ") and line != b"data: [DONE]"]
        calls, finish, text = {}, None, ""
        for event in events:
            for choice in event.get("choices", []):
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta", {})
                text += delta.get("content") or ""
                for call in delta.get("tool_calls") or []:
                    current = calls.setdefault(call["index"], {"id": None, "name": "", "arguments": ""})
                    current["id"] = call.get("id") or current["id"]
                    fn = call.get("function") or {}
                    current["name"] += fn.get("name") or ""
                    current["arguments"] += fn.get("arguments") or ""
        calls = list(calls.values())
    else:
        choice = json.loads(wire)["choices"][0]
        finish = choice["finish_reason"]
        message = choice["message"]
        text = message.get("content") or ""
        calls = [{"id": c["id"], **c["function"]} for c in message.get("tool_calls") or []]
    assert finish == "tool_calls", {"finish": finish, "content": text}
    assert len(calls) == 1, calls
    assert calls[0]["id"] and calls[0]["name"] == "cascade__world_state", calls
    assert json.loads(calls[0]["arguments"]) == {}
    assert "<tool_call>" not in text and "<function=" not in text, text
