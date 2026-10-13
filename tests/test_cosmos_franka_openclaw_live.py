"""Opt-in BF16 generation with CASCADE's real read-only MCP tool schemas.

No MCP tool is executed and no camera/robot connection is created. Native HTTP
and SSE results establish protocol compatibility, not physical readiness.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

import pytest

from test_cosmos_live import infer, native_call


SYSTEM = (
    "You are a robot assistant. Use the available tools to inspect the physical "
    "workspace when asked. Observations must come from tools."
)


def inspection_tools():
    # Parse data only: importing/constructing the MCP server is unnecessary.
    path = Path(__file__).resolve().parents[1] / "src/cascade/apps/mcp_server.py"
    tree = ast.parse(path.read_text())
    specs = next(ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and any(
                     isinstance(target, ast.Name) and target.id == "_EXTRA_TOOLS"
                     for target in node.targets))
    return [{"type": "function", "function": dict(spec, name="cascade__" + spec["name"])}
            for spec in specs if spec["name"] in {"world_state", "camera_snapshot"}]


def request_for(prompt):
    return {
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
        "tools": inspection_tools(), "tool_choice": "auto",
    }


@pytest.mark.parametrize("label,prompt,expected", [
    ("world_state", "What objects does the tracked world model currently contain?", {}),
    ("camera_snapshot", "Show me the d435i camera.", {"camera": "d435i"}),
])
def test_live_cosmos_actual_mcp_inspection_tool(label, prompt, expected):
    if not os.environ.get("COSMOS_TEST_BASE_URL"):
        pytest.skip("requires explicit live Cosmos loopback endpoint")
    result = infer("franka-mcp-" + label, request_for(prompt))
    assert native_call(result, "cascade__" + label) == expected


def test_live_cosmos_actual_mcp_camera_stream():
    base = os.environ.get("COSMOS_TEST_BASE_URL")
    if not base:
        pytest.skip("requires explicit live Cosmos loopback endpoint")
    assert urlsplit(base).hostname in {"127.0.0.1", "localhost", "::1"}
    payload = dict(request_for("Show me the d435i camera."), model="cosmos3-edge",
                   stream=True, temperature=0, max_tokens=256,
                   chat_template_kwargs={"enable_thinking": False})
    events, calls, finish = [], {}, None
    opener = build_opener(ProxyHandler({}))
    request = Request(base.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=120) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            if line.strip() == b"data: [DONE]":
                break
            chunk = json.loads(line[6:])
            events.append(chunk)
            for choice in chunk.get("choices", []):
                finish = choice.get("finish_reason") or finish
                for delta in choice["delta"].get("tool_calls") or []:
                    current = calls.setdefault(delta["index"], {
                        "id": None, "type": "function", "function": {"name": "", "arguments": ""},
                    })
                    current["id"] = delta.get("id") or current["id"]
                    for field in ("name", "arguments"):
                        current["function"][field] += (delta.get("function") or {}).get(field) or ""
    if directory := os.environ.get("COSMOS_TEST_EVIDENCE_DIR"):
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "franka-mcp-camera-stream.json").write_text(
            json.dumps({"request": payload, "events": events}, indent=2))
    result = {"finish_reason": finish, "message": {"tool_calls": list(calls.values())}}
    assert native_call(result, "cascade__camera_snapshot") == {"camera": "d435i"}
