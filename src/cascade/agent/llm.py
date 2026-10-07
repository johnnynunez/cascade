"""LLM backends behind one tool-calling chat interface.

- OpenAICompatClient: any OpenAI-compatible server. This is the single path
  for BOTH cloud OpenAI-style APIs and the local DGX Spark stack (llama.cpp
  llama-server / vLLM serving Qwen3.6 with MTP speculative decoding expose
  the same /v1/chat/completions API).
- AnthropicClient: Claude over the Anthropic API (tool use blocks).
- CodexExecClient: GPT-6-Astra (or any Codex model) THROUGH the Codex CLI
  (`codex exec`) on a ChatGPT/Codex subscription -- no API key, no HTTP from
  this process; one subprocess per step with a strict JSON output schema.
- MockLLM: scripted responses for tests and --llm mock dry-runs.

Images ride along as base64 JPEG (data URLs for OpenAI-compat, source blocks
for Anthropic, `-i` files for Codex); backends that are text-only simply
ignore them.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Cfg

log = logging.getLogger(__name__)


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

    def close(self) -> None:
        """Release this client's owned IO after its last chat call has returned."""

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
        api_key_env: str | None = None,
        supports_vision: bool = True,
        temperature: float = 0.2,
    ):
        import os

        from openai import OpenAI

        # A profile may name its own key variable (`api_key_env`), because
        # OpenAI-compatible does not mean OpenAI-keyed: Nous Portal reads
        # NOUS_API_KEY, and hardcoding OPENAI_API_KEY here would silently send
        # an empty credential to every non-OpenAI provider.
        if api_key is None and api_key_env:
            api_key = os.environ.get(api_key_env) or None
            if api_key is None:
                raise RuntimeError(
                    f"{api_key_env} is not set. This profile authenticates with "
                    f"it; export the key or pick another --llm profile."
                )
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
        extra_body = self._extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
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

    def _extra_body(self) -> dict:
        """Provider-specific HTTP fields; empty for standard OpenAI."""
        return {}

    def close(self) -> None:
        self._client.close()

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

    def close(self) -> None:
        self._client.close()

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


# ── Codex CLI: GPT-6-Astra on a ChatGPT/Codex subscription ────────────────

CODEX_BIN_ENV = "CASCADE_CODEX_BIN"
CODEX_HOME_ENV = "CODEX_HOME"

#: Final-message shape enforced through `codex exec --output-schema`. OpenAI's
#: strict mode REJECTS free-form objects ("properties is required for object
#: schemas"), so the tool arguments travel as a JSON-encoded string that the
#: client decodes.
CODEX_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments_json": {"type": "string"},
                },
                "required": ["name", "arguments_json"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["text", "tool_calls"],
    "additionalProperties": False,
}

#: Codex feature flags passed as `--enable`. `skip_host_skill_discovery`
#: (codex-cli 0.160: "under development", accepted with a warning) keeps the
#: user's ~/.codex/skills out of the brain prompt -- without it Codex injects
#: skill descriptions until "Exceeded skills context budget" (measured with
#: 660 installed skills). A profile may set `codex_features: []` if a future
#: CLI rejects it.
CODEX_DEFAULT_FEATURES: tuple[str, ...] = ("skip_host_skill_discovery",)

_CODEX_TOOL_RULES = (
    "You act through the tools below. Choose at most ONE tool call per turn, "
    "or none when you answer in text. You cannot run commands or read files "
    "yourself: the harness executes the call and returns the result as the "
    "next tool-result message. Never call a tool that is not listed."
)

_CODEX_OUTPUT_RULES = (
    "Reply with exactly one JSON object of the form "
    '{"text": string, "tool_calls": [{"name": string, "arguments_json": string}]}. '
    '"text" is what you want to say (may be empty). "tool_calls" holds at most '
    'one entry; "arguments_json" is the tool\'s arguments object encoded as a '
    "JSON string (\"{}\" when the tool takes none). Any content outside that "
    "JSON object is discarded."
)


def codex_home() -> Path:
    """`$CODEX_HOME`, else `~/.codex` -- the same resolution the CLI uses."""
    return Path(os.environ.get(CODEX_HOME_ENV, "").strip() or Path.home() / ".codex")


def codex_auth_present() -> bool:
    """True when the Codex CLI is logged in: `$CODEX_HOME/auth.json` carries a
    ChatGPT access token (`tokens.access_token`) or an API key stored by
    `codex login --with-api-key` (`OPENAI_API_KEY`). Returns a bool only; the
    values are never logged, returned or copied."""
    try:
        data = json.loads((codex_home() / "auth.json").read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    tokens = data.get("tokens")
    if isinstance(tokens, dict) and str(tokens.get("access_token") or "").strip():
        return True
    return bool(str(data.get("OPENAI_API_KEY") or "").strip())


def codex_binary(codex_bin: str | None = None) -> str | None:
    """Resolve the runnable codex executable: `CASCADE_CODEX_BIN`, then the
    profile's `codex_bin`, then `codex` on PATH. None when nothing runs."""
    import shutil

    candidate = os.environ.get(CODEX_BIN_ENV, "").strip() or (codex_bin or "codex")
    return shutil.which(candidate)


def _tail(path: Path, n: int = 20) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


class CodexExecClient(LLMClient):
    """GPT-6-Astra as the brain, THROUGH the Codex CLI.

    The user's Codex subscription has no OPENAI_API_KEY, so api.openai.com is
    not reachable from this process: the only way to the model is `codex
    exec`. Each chat() call renders the whole exchange -- system text, tool
    specs, the conversation with tool results verbatim and prior assistant
    calls, the output rules -- into ONE prompt, runs

        codex exec --ignore-user-config --ephemeral --skip-git-repo-check
                   -s read-only -C <empty tmp dir> -m <model>
                   -c model_reasoning_effort=<effort> --enable <feature>...
                   --output-schema schema.json -o out.json --color never
                   [-i img.jpg ...] -       # prompt on stdin

    and parses the final message (schema-enforced JSON) into an LLMResponse.
    `--ignore-user-config` matters: the user's ~/.codex/config.toml registers
    MCP servers, and a brain whose Codex session could call the robot MCP
    server directly would bypass CASCADE's harness. `-C` points at a private
    empty directory so no repo AGENTS.md leaks into the prompt. Only the
    images of the LAST image-carrying message (the current observation and
    its memory frames) are attached; older ones are replaced by a marker.
    Measured 2026-10-07: 7-9 s per step at medium reasoning effort.
    `max_tokens` has no Codex equivalent and is ignored. Nothing here reads
    or forwards an API key; `OPENAI_API_KEY` is stripped from the child's
    environment so a stray export cannot move the brain off the subscription.
    """

    def __init__(
        self,
        model: str = "gpt-6-astra",
        reasoning_effort: str = "medium",
        codex_bin: str | None = None,
        supports_vision: bool = True,
        timeout_s: float = 240.0,
        codex_config: list[str] | tuple[str, ...] | None = None,
        codex_features: list[str] | tuple[str, ...] | None = CODEX_DEFAULT_FEATURES,
        max_images: int | None = None,
    ):
        resolved = codex_binary(codex_bin)
        if resolved is None:
            wanted = os.environ.get(CODEX_BIN_ENV, "").strip() or (codex_bin or "codex")
            raise RuntimeError(
                f"codex CLI not found ({wanted!r}). Install the Codex CLI, point "
                f"{CODEX_BIN_ENV} at it, or pick another --llm profile."
            )
        if not codex_auth_present():
            raise RuntimeError(
                f"the Codex CLI is not logged in: no access token in "
                f"{codex_home() / 'auth.json'}. Run `codex login` (ChatGPT/Codex "
                f"subscription) or pick another --llm profile."
            )
        self.codex_bin = resolved
        self.model = model
        self.reasoning_effort = str(reasoning_effort)
        self.supports_vision = bool(supports_vision)
        self.timeout_s = float(timeout_s)
        self.codex_config = [str(c) for c in (codex_config or ())]
        self.codex_features = [str(f) for f in (codex_features or ())]
        self.max_images = None if max_images is None else int(max_images)
        self._calls = 0
        self._tmp_root: str | None = None

    # ── temp files ───────────────────────────────────────────────────────

    @property
    def temp_root(self) -> str | None:
        """Directory holding this client's per-step files (None before use)."""
        return self._tmp_root

    def _ensure_root(self) -> Path:
        import tempfile

        if self._tmp_root is None or not os.path.isdir(self._tmp_root):
            self._tmp_root = tempfile.mkdtemp(prefix="cascade-codex-")
        return Path(self._tmp_root)

    def close(self) -> None:
        import shutil

        if self._tmp_root is not None:
            shutil.rmtree(self._tmp_root, ignore_errors=True)
            self._tmp_root = None

    # ── rendering ────────────────────────────────────────────────────────

    def _select_images(self, messages: list[dict]) -> tuple[int | None, list[bytes], int]:
        """(index of the message whose images are attached, those images,
        how many of that message's images were dropped by max_images)."""
        last = None
        for i, m in enumerate(messages):
            if m.get("images"):
                last = i
        if last is None or not self.supports_vision:
            return None, [], 0
        images = list(messages[last]["images"])
        if self.max_images is not None and len(images) > self.max_images:
            dropped = len(images) - self.max_images
            # the orchestrator puts the current view LAST: keep the newest
            images = images[dropped:]
            return last, images, dropped
        return last, images, 0

    def _render_prompt(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None,
        attached_at: int | None,
        n_attached: int,
        n_dropped: int,
    ) -> str:
        parts: list[str] = [system.strip(), ""]
        if tools:
            parts += ["# Tools", _CODEX_TOOL_RULES]
            for t in tools:
                params = t.get("parameters", {"type": "object", "properties": {}})
                parts.append(f"- {t['name']}: {t.get('description', '').strip()}")
                parts.append(f"  parameters: {json.dumps(params, ensure_ascii=False)}")
            parts.append("")
        parts.append("# Conversation")
        for i, m in enumerate(messages):
            role = m.get("role", "user")
            if role == "tool":
                parts.append(f"[tool result: {m.get('name', '?')} ({m.get('tool_call_id', 'call_0')})]")
                content = m.get("content", "")
                parts.append(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
                continue
            parts.append(f"[{role}]")
            content = m.get("content") or ""
            if content:
                parts.append(content if isinstance(content, str) else str(content))
            for tc in m.get("tool_calls") or []:
                parts.append(
                    f"called {tc.name} {json.dumps(tc.arguments, ensure_ascii=False, default=str)}"
                )
            images = m.get("images") or []
            if images:
                if i == attached_at and n_attached:
                    note = (
                        f"[{n_attached} image(s) attached to this message, in this order; "
                        f"the last one is the newest]"
                    )
                    if n_dropped:
                        note += f" [{n_dropped} earlier image(s) omitted]"
                    parts.append(note)
                else:
                    parts.extend(["[image omitted]"] * len(images))
        parts += ["", "# Output", _CODEX_OUTPUT_RULES, ""]
        return "\n".join(parts)

    # ── the subprocess ───────────────────────────────────────────────────

    def _command(self, cwd: Path, schema: Path, out: Path, images: list[Path]) -> list[str]:
        cmd = [
            self.codex_bin, "exec",
            "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
            "-s", "read-only",
            "-C", str(cwd),
            "-m", self.model,
            "-c", f"model_reasoning_effort={self.reasoning_effort}",
        ]
        for feature in self.codex_features:
            cmd += ["--enable", feature]
        for kv in self.codex_config:
            cmd += ["-c", kv]
        cmd += ["--output-schema", str(schema), "-o", str(out), "--color", "never"]
        for img in images:
            cmd += ["-i", str(img)]
        cmd.append("-")  # the prompt arrives on stdin: transcripts outgrow argv
        return cmd

    @staticmethod
    def _kill(proc) -> None:
        import signal

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 - best effort reap
            pass

    def chat(self, system, messages, tools=None, max_tokens=1024) -> LLMResponse:
        import shlex
        import shutil
        import subprocess
        import tempfile
        import time

        step = self._calls
        self._calls += 1
        call_dir = Path(tempfile.mkdtemp(prefix=f"step{step:04d}-", dir=self._ensure_root()))
        try:
            cwd = call_dir / "cwd"  # the brain's working root: private and EMPTY
            cwd.mkdir()
            attached_at, jpegs, dropped = self._select_images(messages)
            image_paths: list[Path] = []
            for k, jpeg in enumerate(jpegs):
                p = call_dir / f"img{k:02d}.jpg"
                p.write_bytes(jpeg)
                image_paths.append(p)
            prompt = self._render_prompt(system, messages, tools, attached_at, len(jpegs), dropped)
            prompt_path = call_dir / "prompt.txt"
            prompt_path.write_text(prompt)
            schema_path = call_dir / "schema.json"
            schema_path.write_text(json.dumps(CODEX_OUTPUT_SCHEMA))
            out_path = call_dir / "out.json"
            stdout_path = call_dir / "stdout.txt"
            stderr_path = call_dir / "stderr.txt"
            cmd = self._command(cwd, schema_path, out_path, image_paths)
            env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
            # Operators read this from the run log: which exact command a step
            # ran (model, effort, isolation flags). Temp paths only, no secrets.
            log.info("codex exec argv (step %d, %d image(s)): %s", step, len(jpegs), shlex.join(cmd))

            t0 = time.monotonic()
            with open(prompt_path, "rb") as stdin_f, open(stdout_path, "wb") as out_f, \
                    open(stderr_path, "wb") as err_f:
                proc = subprocess.Popen(
                    cmd, stdin=stdin_f, stdout=out_f, stderr=err_f,
                    cwd=str(cwd), env=env, start_new_session=True,
                )
                try:
                    rc = proc.wait(timeout=self.timeout_s)
                except subprocess.TimeoutExpired:
                    self._kill(proc)
                    raise RuntimeError(
                        f"codex exec timed out after {self.timeout_s:.0f} s "
                        f"(model {self.model}, step {step}); process group killed. "
                        f"stderr tail:\n{_tail(stderr_path)}"
                    ) from None
            dt = time.monotonic() - t0
            if rc != 0:
                raise RuntimeError(
                    f"codex exec exited with status {rc} (model {self.model}, step {step}). "
                    f"stderr tail:\n{_tail(stderr_path)}"
                )
            try:
                raw = out_path.read_text()
            except OSError:
                raw = ""
            if not raw.strip():
                raise RuntimeError(
                    f"codex exec wrote no final message (model {self.model}, step {step}). "
                    f"stderr tail:\n{_tail(stderr_path)}"
                )
            resp = self._parse(raw, stderr_path, step)
            log.info(
                "codex exec: %s/%s step %d, %d image(s), %.1f s -> %s",
                self.model, self.reasoning_effort, step, len(jpegs), dt,
                resp.tool_calls[0].name if resp.tool_calls else "text",
            )
            return resp
        finally:
            shutil.rmtree(call_dir, ignore_errors=True)

    def _parse(self, raw: str, stderr_path: Path, step: int) -> LLMResponse:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"could not parse the codex final message as JSON ({e}): {raw[:300]!r}. "
                f"stderr tail:\n{_tail(stderr_path)}"
            ) from None
        if not isinstance(data, dict):
            raise RuntimeError(
                f"could not parse the codex final message: expected an object, got "
                f"{type(data).__name__}: {raw[:300]!r}"
            )
        text = data.get("text")
        text = text if isinstance(text, str) else ("" if text is None else str(text))
        calls: list[ToolCall] = []
        raw_calls = data.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raw_calls = []
        if len(raw_calls) > 1:
            log.warning("codex exec returned %d tool calls; keeping the first", len(raw_calls))
        for tc in raw_calls[:1]:
            if not isinstance(tc, dict) or not tc.get("name"):
                continue
            args_json = tc.get("arguments_json")
            if isinstance(args_json, dict):  # a lenient model may inline the object
                args: dict[str, Any] = args_json
            else:
                args_json = "" if args_json is None else str(args_json)
                try:
                    parsed = json.loads(args_json or "{}")
                except json.JSONDecodeError:
                    parsed = None
                args = parsed if isinstance(parsed, dict) else {"_raw": args_json}
            calls.append(ToolCall(name=str(tc["name"]), arguments=args, id=f"call_{step}"))
        return LLMResponse(text=text, tool_calls=calls)


#: The profile `--llm auto` picks FIRST, before any API-key profile, when the
#: Codex CLI is installed and logged in (user decision 2026-10-07: GPT-6-Astra
#: through the Codex subscription is the brain; the local Qwen3 was judged too
#: weak). `CASCADE_LLM=<profile>` or an explicit `--llm <profile>` still pins.
AUTO_LLM_CODEX_PROFILE = "codex_astra"


#: API-key brain profiles `--llm auto` will pick, in order, when their key is
#: present -- after the Codex login check above. Hermes/Nous Portal leads this
#: table: it is the project's default host (see configs/llm/hermes.yaml and
#: scripts/setup_agents.py) and one Portal subscription covers the whole model
#: range, so it is the fewest-steps path from a fresh clone to a real brain.
AUTO_LLM_PROFILES: tuple[tuple[str, str], ...] = (
    ("hermes", "NOUS_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("openai", "OPENAI_API_KEY"),
)

LLM_ENV = "CASCADE_LLM"


def resolve_llm_profile(name: str | None = "auto") -> str:
    """`--llm` value -> a profile name that can actually be constructed here.

    `auto` exists so a fresh clone runs offline AND a configured machine gets a
    real brain from the same command. Flipping the default outright to `hermes`
    would make the documented offline wiring check (`python -m cascade.apps.demo
    --task ...`) fail on any machine without credentials, which is a bad trade
    for a framework whose mock stack is the thing you try first.

    Order: a logged-in Codex CLI (GPT-6-Astra on the subscription) first, then
    the API-key profiles in `AUTO_LLM_PROFILES`, then mock. A Codex login whose
    CLI is not runnable (no `codex` on PATH, no CASCADE_CODEX_BIN) is skipped
    rather than turned into a construction error.

    Same degrade-the-capability-never-the-session rule the grasp, occupancy and
    device layers follow.
    """
    env = os.environ.get(LLM_ENV, "").strip()
    requested = (env or name or "auto").strip()
    if requested.lower() != "auto":
        return requested
    if codex_auth_present():
        if codex_binary() is not None:
            log.info("llm: auto -> %s (Codex CLI is logged in)", AUTO_LLM_CODEX_PROFILE)
            return AUTO_LLM_CODEX_PROFILE
        log.info(
            "llm: Codex login found but no runnable codex CLI (PATH or %s); skipping %s",
            CODEX_BIN_ENV, AUTO_LLM_CODEX_PROFILE,
        )
    for profile, key in AUTO_LLM_PROFILES:
        if os.environ.get(key, "").strip():
            log.info("llm: auto -> %s (%s is set)", profile, key)
            return profile
    log.info(
        "llm: auto -> mock (no brain credentials found; `codex login` or set one of %s)",
        ", ".join(k for _, k in AUTO_LLM_PROFILES),
    )
    return "mock"


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
            api_key_env=cfg.get("api_key_env"),
            supports_vision=bool(cfg.get("supports_vision", True)),
            temperature=float(cfg.get("temperature", 0.2)),
        )
    if kind == "cosmos3":
        # NVIDIA Cosmos3-Edge speaks OpenAI-compatible HTTP but emits XML
        # tool calls, so it needs its own response parser (see agent/cosmos3).
        from .cosmos3 import make_cosmos3_client

        return make_cosmos3_client(cfg)
    if kind == "codex_exec":
        features = cfg.get("codex_features", list(CODEX_DEFAULT_FEATURES))
        return CodexExecClient(
            model=cfg.get("model", "gpt-6-astra"),
            reasoning_effort=str(cfg.get("reasoning_effort", "medium")),
            codex_bin=cfg.get("codex_bin"),
            supports_vision=bool(cfg.get("supports_vision", True)),
            timeout_s=float(cfg.get("timeout_s", 240)),
            codex_config=list(cfg.get("codex_config") or []),
            codex_features=list(features or []),
            max_images=cfg.get("max_images"),
        )
    raise ValueError(
        f"unknown llm type {kind!r} (mock|anthropic|openai_compat|cosmos3|codex_exec)"
    )
