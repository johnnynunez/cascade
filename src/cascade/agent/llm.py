"""LLM backends behind one tool-calling chat interface.

- OpenAICompatClient: any OpenAI-compatible server. This is the single path
  for BOTH cloud OpenAI-style APIs and the local DGX Spark stack (llama.cpp
  llama-server / vLLM serving Qwen3.6 with MTP speculative decoding expose
  the same /v1/chat/completions API).
- AnthropicClient: Claude over the Anthropic API (tool use blocks).
- MockLLM: scripted responses for tests and --llm mock dry-runs.

Images ride along as base64 JPEG (data URLs for OpenAI-compat, source blocks
for Anthropic); backends that are text-only simply ignore them.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any

from ..config import Cfg


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient:
    supports_vision = False

    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """messages: [{role, content:str, images?: [jpeg bytes]}, ...] where a
        role="tool" message carries {tool_call_id, name, content}."""
        raise NotImplementedError


def _b64(jpeg: bytes) -> str:
    return base64.b64encode(jpeg).decode()


class OpenAICompatClient(LLMClient):
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        supports_vision: bool = True,
        temperature: float = 0.2,
    ):
        import os

        from openai import OpenAI

        # Local servers (llama.cpp/vLLM) need a dummy key; hosted endpoints
        # fall through to the OPENAI_API_KEY environment variable.
        if api_key is None and base_url is not None and "OPENAI_API_KEY" not in os.environ:
            api_key = "EMPTY"
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.supports_vision = supports_vision
        self.temperature = temperature

    def chat(self, system, messages, tools=None, max_tokens=1024) -> LLMResponse:
        oai_msgs: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "tool":
                oai_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.get("tool_call_id", "call_0"),
                        "content": m["content"],
                    }
                )
                continue
            if m["role"] == "assistant" and m.get("tool_calls"):
                oai_msgs.append(
                    {
                        "role": "assistant",
                        "content": m.get("content") or None,
                        "tool_calls": [
                            {
                                "id": tc.id or f"call_{i}",
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for i, tc in enumerate(m["tool_calls"])
                        ],
                    }
                )
                continue
            images = m.get("images") or []
            if images and self.supports_vision:
                content: Any = [{"type": "text", "text": m["content"]}]
                for jpeg in images:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{_b64(jpeg)}"},
                        }
                    )
            else:
                content = m["content"]
            oai_msgs.append({"role": m["role"], "content": content})

        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=oai_msgs,
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": t} for t in tools
            ]
            # One tool call per turn keeps the loop deterministic.
            kwargs["parallel_tool_calls"] = False
        resp = self._call_with_param_fallback(kwargs)
        choice = resp.choices[0].message
        calls = []
        for tc in choice.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(name=tc.function.name, arguments=args, id=tc.id))
        return LLMResponse(text=choice.content or "", tool_calls=calls)

    def _call_with_param_fallback(self, kwargs: dict):
        """Newer OpenAI models reject max_tokens (want max_completion_tokens);
        some local servers reject parallel_tool_calls. Retry without the
        offending parameter instead of failing the demo."""
        for _ in range(3):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as e:
                msg = str(e)
                if "max_completion_tokens" in msg and "max_tokens" in kwargs:
                    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                    continue
                if "parallel_tool_calls" in msg and "parallel_tool_calls" in kwargs:
                    kwargs.pop("parallel_tool_calls")
                    continue
                if "temperature" in msg and "temperature" in kwargs:
                    kwargs.pop("temperature")
                    continue
                raise
        return self._client.chat.completions.create(**kwargs)


class AnthropicClient(LLMClient):
    supports_vision = True

    def __init__(self, model: str = "claude-sonnet-5", temperature: float = 0.2):
        import anthropic

        self._client = anthropic.Anthropic()
        self.model = model
        self.temperature = temperature

    def chat(self, system, messages, tools=None, max_tokens=1024) -> LLMResponse:
        anth_msgs: list[dict] = []
        for m in messages:
            if m["role"] == "tool":
                anth_msgs.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.get("tool_call_id", "call_0"),
                                "content": m["content"],
                            }
                        ],
                    }
                )
                continue
            if m["role"] == "assistant" and m.get("tool_calls"):
                blocks: list[dict] = []
                if (m.get("content") or "").strip():
                    blocks.append({"type": "text", "text": m["content"]})
                for i, tc in enumerate(m["tool_calls"]):
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id or f"call_{i}",
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                anth_msgs.append({"role": "assistant", "content": blocks})
                continue
            # The API rejects empty text blocks; substitute a placeholder.
            text = m["content"] if (m.get("content") or "").strip() else "(no text)"
            content: list[dict] = [{"type": "text", "text": text}]
            for jpeg in m.get("images") or []:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": _b64(jpeg),
                        },
                    }
                )
            anth_msgs.append({"role": m["role"], "content": content})

        kwargs: dict[str, Any] = dict(
            model=self.model,
            system=system,
            messages=anth_msgs,
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        if tools:
            kwargs["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {"type": "object"}),
                }
                for t in tools
            ]
            kwargs["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        resp = self._client.messages.create(**kwargs)
        text_parts, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(name=block.name, arguments=dict(block.input), id=block.id))
        return LLMResponse(text="\n".join(text_parts), tool_calls=calls)


class MockLLM(LLMClient):
    """Pops scripted responses; records every request for assertions."""

    supports_vision = True

    def __init__(self, script: list[LLMResponse]):
        self._script = list(script)
        self.requests: list[dict] = []

    def chat(self, system, messages, tools=None, max_tokens=1024) -> LLMResponse:
        # Snapshot the (mutable, reused) message list so per-request
        # assertions in tests see what was actually sent at that time.
        self.requests.append(
            {"system": system, "messages": [dict(m) for m in messages], "tools": tools}
        )
        if not self._script:
            return LLMResponse(text="(mock script exhausted)", tool_calls=[])
        return self._script.pop(0)


def make_llm(cfg: Cfg) -> LLMClient:
    kind = cfg.type
    if kind == "mock":
        # Default wiring-check script: observe once, then finish honestly.
        return MockLLM(
            [
                LLMResponse(tool_calls=[ToolCall("get_observation", {})]),
                LLMResponse(
                    tool_calls=[
                        ToolCall(
                            "task_done",
                            {
                                "success": True,
                                "summary": "mock LLM wiring check: observed the scene and stopped",
                            },
                        )
                    ]
                ),
            ]
        )
    if kind == "anthropic":
        return AnthropicClient(
            model=cfg.get("model", "claude-sonnet-5"),
            temperature=float(cfg.get("temperature", 0.2)),
        )
    if kind == "openai_compat":
        return OpenAICompatClient(
            model=cfg.model,
            base_url=cfg.get("base_url"),
            api_key=cfg.get("api_key"),
            supports_vision=bool(cfg.get("supports_vision", True)),
            temperature=float(cfg.get("temperature", 0.2)),
        )
    if kind == "cosmos3":
        # NVIDIA Cosmos3-Edge speaks OpenAI-compatible HTTP but emits XML
        # tool calls, so it needs its own response parser (see agent/cosmos3).
        from .cosmos3 import make_cosmos3_client

        return make_cosmos3_client(cfg)
    raise ValueError(f"unknown llm type {kind!r} (mock|anthropic|openai_compat|cosmos3)")
