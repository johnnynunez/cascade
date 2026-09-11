"""Opt-in real Cosmos HTTP inference. Never a parser fixture or robot action.

COSMOS_TEST_BASE_URL=http://127.0.0.1:8082/v1 enables live GPU requests.
COSMOS_TEST_EVIDENCE_DIR optionally saves exact requests/responses as receipts.
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

import pytest


def infer(label, payload):
    base = os.environ.get("COSMOS_TEST_BASE_URL")
    if not base:
        pytest.skip("requires explicit live Cosmos loopback endpoint")
    parsed = urlsplit(base)
    assert parsed.hostname in {"127.0.0.1", "localhost", "::1"}, "live probe must stay local"
    request = dict(model="cosmos3-edge", stream=False, temperature=0, max_tokens=128,
                   chat_template_kwargs={"enable_thinking": False}, **payload)
    start = time.time()
    opener = build_opener(ProxyHandler({}))
    with opener.open(Request(base.rstrip("/") + "/chat/completions",
                             data=json.dumps(request).encode(),
                             headers={"Content-Type": "application/json"}), timeout=120) as result:
        response = json.loads(result.read())
    if dest := os.environ.get("COSMOS_TEST_EVIDENCE_DIR"):
        path = Path(dest)
        path.mkdir(parents=True, exist_ok=True)
        (path / (label + ".json")).write_text(json.dumps(
            {"base_url": base, "started_unix": start, "elapsed_s": time.time() - start,
             "request": request, "response": response}, indent=2))
    assert response["model"] == "cosmos3-edge"
    assert response["usage"]["completion_tokens"] > 0
    return response["choices"][0]


def native_call(choice, name):
    assert choice["finish_reason"] == "tool_calls", choice
    message = choice["message"]
    calls = message.get("tool_calls")
    assert isinstance(calls, list) and len(calls) == 1, message
    call = calls[0]
    assert call["id"] and call["type"] == "function"
    assert call["function"]["name"] == name
    assert "<tool_call>" not in (message.get("content") or "")
    return json.loads(call["function"]["arguments"])


def test_live_cosmos_text():
    result = infer("text", {"messages": [{"role": "user", "content": "Reply with exactly COSMOS_TEXT_OK."}]})
    assert result["finish_reason"] == "stop", result
    assert result["message"]["content"].strip() == "COSMOS_TEXT_OK"
    assert not result["message"].get("tool_calls")


def test_live_cosmos_native_readiness():
    name = "cascade_readiness"
    result = infer("native-readiness", {
        "messages": [{"role": "user", "content": "Call cascade_readiness once with ready=true. Do not answer in prose."}],
        "tools": [{"type": "function", "function": {"name": name,
            "description": "Read-only protocol probe; no side effects.",
            "parameters": {"type": "object", "properties": {"ready": {"type": "boolean"}}, "required": ["ready"]}}}],
        "tool_choice": "auto",
    })
    assert native_call(result, name) == {"ready": True}


def test_live_cosmos_streamed_native_namespaced_arguments():
    base = os.environ.get("COSMOS_TEST_BASE_URL")
    if not base:
        pytest.skip("requires explicit live Cosmos loopback endpoint")
    assert urlsplit(base).hostname in {"127.0.0.1", "localhost", "::1"}
    name = "cascade__inspect_scene"
    expected = {"enabled": False, "count": 3, "point": [0.1, 0.2], "label": "sample"}
    payload = {"model": "cosmos3-edge", "stream": True, "temperature": 0, "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": "Call cascade__inspect_scene once with enabled=false, count=3, point=[0.1,0.2], label=sample. Do not answer in prose."}],
        "tools": [{"type": "function", "function": {"name": name,
            "description": "Read-only protocol probe; no side effects.",
            "parameters": {"type": "object", "properties": {
                "enabled": {"type": "boolean"}, "count": {"type": "integer"},
                "point": {"type": "array", "items": {"type": "number"}},
                "label": {"type": "string"}}, "required": list(expected)}}}],
        "tool_choice": "auto"}
    opener = build_opener(ProxyHandler({}))
    events, calls, finish = [], {}, None
    with opener.open(Request(base.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"}), timeout=120) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            if line.strip() == b"data: [DONE]":
                break
            chunk = json.loads(line[6:])
            events.append(chunk)
            for choice in chunk.get("choices", []):
                finish = choice.get("finish_reason") or finish
                for call in choice["delta"].get("tool_calls") or []:
                    current = calls.setdefault(call["index"], {"id": None, "name": "", "arguments": ""})
                    current["id"] = call.get("id") or current["id"]
                    fn = call.get("function") or {}
                    current["name"] += fn.get("name") or ""
                    current["arguments"] += fn.get("arguments") or ""
    if dest := os.environ.get("COSMOS_TEST_EVIDENCE_DIR"):
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "native-stream.json").write_text(json.dumps({"request": payload, "events": events}, indent=2))
    assert finish == "tool_calls" and len(calls) == 1, (finish, calls)
    call = next(iter(calls.values()))
    assert call["id"] and call["name"] == name
    assert json.loads(call["arguments"]) == expected


@pytest.mark.parametrize("color,rgb", [("red", (255, 0, 0)), ("blue", (0, 0, 255))])
def test_live_cosmos_image_controls_native_namespaced_tool_arguments(color, rgb):
    if not os.environ.get("COSMOS_TEST_BASE_URL"):
        pytest.skip("requires explicit live Cosmos loopback endpoint")
    from PIL import Image

    # Controlled input image, NOT a fabricated inference response. The same
    # text contains neither expected color; only pixels distinguish the cases.
    image = Image.new("RGB", (320, 256), rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    name = "cascade__report_color"
    result = infer("image-native-" + color, {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "Use cascade__report_color once to report the dominant color actually visible in the attached image, with observed=true. Do not answer in prose."},
        ]}],
        "tools": [{"type": "function", "function": {"name": name,
            "description": "Read-only visual observation; no robot action.",
            "parameters": {"type": "object", "properties": {
                "color": {"type": "string", "description": "One lowercase English color name."},
                "observed": {"type": "boolean"}}, "required": ["color", "observed"]}}}],
        "tool_choice": "auto",
    })
    assert native_call(result, name) == {"color": color, "observed": True}
