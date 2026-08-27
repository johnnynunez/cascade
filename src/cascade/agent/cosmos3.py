"""NVIDIA Cosmos3-Edge backend: the reasoner tower as the demo's agent brain.

Cosmos3 is an omnimodal world model built as a Mixture-of-Transformers: an
autoregressive tower for text and a diffusion tower for images/video/actions.
The **Edge** checkpoint is the 4B member of the family -- small enough to sit
beside Isaac Sim on one RTX PRO 6000 while the sim owns most of the VRAM,
which is exactly the constraint this demo runs under (Qwen3-VL Q4_K_M costs
~18 GB; Cosmos3-Edge in BF16 costs ~8.6 GB and is natively multimodal, so no
separate mmproj projector).

Why it belongs here specifically:

* It is trained for **Physical AI** -- the reasoner is evaluated on embodied
  spatial/temporal reasoning rather than general chat, which is the entire
  job of this demo's advisor and milestone verifier.
* It accepts **video** (recommended 4 fps), not just stills.  The rig already
  runs three cameras through a FrameHub; feeding a short clip lets the model
  reason about *what just happened* (did the cube slip?) instead of guessing
  from one frame.
* 256K context, so ASPIRE library entries + envelope digests + milestone
  boards fit without the pruning gymnastics the Qwen path needs.

**The tool-call catch (this is the whole reason for a dedicated backend).**
Cosmos3-Edge's chat template does NOT emit OpenAI-style JSON ``tool_calls``.
It emits an XML block::

    <tool_call>
    <function=grasp_object>
    <parameter=label>
    pink cube
    </parameter>
    </function>
    </tool_call>

Pointing the existing ``openai_compat`` profile at a Cosmos3 server therefore
"works" -- and silently never calls a single tool, because ``choices[0]
.message.tool_calls`` stays empty and every turn looks like a chat reply.
This client parses both shapes: native ``tool_calls`` when the server has a
matching tool parser plugin, and the XML block out of the text content when it
does not.  Values are coerced against the JSON schema in ``TOOL_SPECS`` so
``distance_m`` arrives as a float, not the string ``"0.08"``.

Serving (see ``scripts/serve_cosmos_vllm.sh``)::

    docker pull vllm/vllm-omni:cosmos3
    vllm serve nvidia/Cosmos3-Edge --omni --host 0.0.0.0 --port 8000 \
        --init-timeout 1800
"""

from __future__ import annotations

import json
import re
from typing import Any

from .llm import LLMResponse, OpenAICompatClient, ToolCall

#: <tool_call> ... </tool_call>, tolerant of whitespace and missing closer
_TOOL_BLOCK = re.compile(r"<tool_call>(.*?)(?:</tool_call>|\Z)", re.S | re.I)
#: <function=name> ... </function>
_FUNCTION = re.compile(r"<function=([A-Za-z0-9_\-]+)\s*>(.*?)(?:</function>|\Z)", re.S | re.I)
#: <parameter=key> value </parameter>
_PARAMETER = re.compile(r"<parameter=([A-Za-z0-9_\-]+)\s*>(.*?)(?:</parameter>|\Z)", re.S | re.I)
#: reasoning trace the template wraps around chain-of-thought
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def _coerce(value: str, spec: dict | None) -> Any:
    """Cast a string parameter to the type its JSON schema declares."""
    text = value.strip()
    kind = (spec or {}).get("type")
    if kind in ("number", "integer"):
        try:
            return int(text) if kind == "integer" else float(text)
        except ValueError:
            return text
    if kind == "boolean":
        return text.strip().lower() in ("true", "yes", "1")
    if kind in ("array", "object"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return [p.strip() for p in text.split(",")] if kind == "array" else text
    if kind is None:
        # No schema: recover obvious JSON, else keep the string.
        if text[:1] in "[{" or text in ("true", "false", "null"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        try:
            return float(text) if "." in text else int(text)
        except ValueError:
            return text
    return text


def parse_xml_tool_calls(text: str, tools: list[dict] | None = None) -> list[ToolCall]:
    """Extract Cosmos3-style XML tool calls from assistant text.

    Returns [] when the text holds no tool block, so callers can fall back to
    treating the reply as prose.
    """
    if not text or "<function=" not in text:
        return []
    schemas: dict[str, dict] = {}
    for spec in tools or []:
        fn = spec.get("function", spec)
        props = ((fn.get("parameters") or {}).get("properties")) or {}
        schemas[fn.get("name", "")] = props

    # Prefer well-formed <tool_call> blocks; fall back to bare <function=...>
    regions = [m.group(1) for m in _TOOL_BLOCK.finditer(text)] or [text]
    calls: list[ToolCall] = []
    for i, region in enumerate(regions):
        for fmatch in _FUNCTION.finditer(region):
            name = fmatch.group(1).strip()
            body = fmatch.group(2)
            props = schemas.get(name, {})
            args: dict[str, Any] = {}
            for pmatch in _PARAMETER.finditer(body):
                key = pmatch.group(1).strip()
                args[key] = _coerce(pmatch.group(2), props.get(key))
            calls.append(ToolCall(name=name, arguments=args, id=f"cosmos_{i}_{len(calls)}"))
    return calls


def strip_reasoning(text: str) -> str:
    """Drop <think> blocks from user-visible narration."""
    return _THINK.sub("", text or "").strip()


class Cosmos3EdgeClient(OpenAICompatClient):
    """OpenAI-compatible transport + Cosmos3 tool-call semantics.

    Inherits the parameter-fallback logic (``max_completion_tokens`` etc.)
    from ``OpenAICompatClient`` and overrides only response interpretation.
    """

    def __init__(
        self,
        model: str = "nvidia/Cosmos3-Edge",
        base_url: str | None = "http://127.0.0.1:8000/v1",
        api_key: str | None = "EMPTY",
        supports_vision: bool = True,
        temperature: float = 0.2,
        enable_thinking: bool = False,
        max_frames: int = 1,
    ):
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            supports_vision=supports_vision,
            temperature=temperature,
        )
        #: The template defaults enable_thinking=True. On a booth clock a
        #: reasoning preamble costs seconds per turn for no visible gain, so
        #: it is off unless explicitly requested.
        self.enable_thinking = bool(enable_thinking)
        #: How many recent frames to attach per turn (the reasoner accepts
        #: video at ~4 fps; >1 lets it see motion, at a token cost).
        self.max_frames = max(1, int(max_frames))

    def chat(self, system, messages, tools=None, max_tokens=1024) -> LLMResponse:
        resp = super().chat(system, messages, tools=tools, max_tokens=max_tokens)
        if resp.tool_calls:
            resp.text = strip_reasoning(resp.text)
            return resp
        calls = parse_xml_tool_calls(resp.text, tools)
        if calls:
            # Keep only the prose before the first tool block as narration.
            head = re.split(r"<tool_call>|<function=", resp.text, maxsplit=1)[0]
            return LLMResponse(text=strip_reasoning(head), tool_calls=calls)
        return LLMResponse(text=strip_reasoning(resp.text), tool_calls=[])

    def _extra_body(self) -> dict:
        return {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}


def make_cosmos3_client(cfg) -> Cosmos3EdgeClient:
    """Build the client from a ``configs/llm/local_cosmos.yaml`` profile."""
    return Cosmos3EdgeClient(
        model=cfg.get("model", "nvidia/Cosmos3-Edge"),
        base_url=cfg.get("base_url", "http://127.0.0.1:8000/v1"),
        api_key=cfg.get("api_key", "EMPTY"),
        supports_vision=bool(cfg.get("supports_vision", True)),
        temperature=float(cfg.get("temperature", 0.2)),
        enable_thinking=bool(cfg.get("enable_thinking", False)),
        max_frames=int(cfg.get("max_frames", 1)),
    )
