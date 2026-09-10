#!/usr/bin/env python3
"""CPU protocol probe: upstream vLLM parser source + official Cosmos tokenizer.

Run in a THROWAWAY environment with transformers==5.17.0, openai and
pydantic. This tokenizer/parser probe does not require torch. Does NOT fetch
checkpoint weights. AST-loads unchanged parser/protocol definitions, excluding
GPU/server imports; this tests parsing, NOT model generation or the HTTP server.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import functools
import hashlib
import json
import logging
import math
from pathlib import Path
import sys
import time
import types
import typing
import urllib.request
import uuid
from dataclasses import dataclass, field
from abc import abstractmethod

VLLM_SHA = "98dff2a81d747d1dba01a47f939f48c3526d4206"  # v0.29.0
MODEL_SHA = "a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba"
SOURCES = {
    "tool_init.py": "vllm/tool_parsers/__init__.py",
    "qwen3.py": "vllm/parser/qwen3.py",
    "abstract_parser.py": "vllm/parser/abstract_parser.py",
    "chat_utils.py": "vllm/entrypoints/chat_utils.py",
    "protocol.py": "vllm/entrypoints/serve/engine/protocol.py",
    "generate_protocol.py": "vllm/entrypoints/generate/base/protocol.py",
    "chat_protocol.py": "vllm/entrypoints/openai/chat_completion/protocol.py",
    "utils.py": "vllm/tool_parsers/utils.py",
}
for _name in (
    "events",
    "incremental_lexer",
    "token_id_scanner",
    "parser_engine_config",
    "streaming_parser_engine",
    "parser_engine",
):
    SOURCES[_name + ".py"] = "vllm/parser/engine/" + _name + ".py"


def source_urls():
    urls = {
        name: f"https://raw.githubusercontent.com/vllm-project/vllm/{VLLM_SHA}/{path}"
        for name, path in SOURCES.items()
    }
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "processor_config.json",
    ):
        urls["hf_" + name] = (
            f"https://huggingface.co/nvidia/Cosmos3-Edge/resolve/{MODEL_SHA}/{name}"
        )
    return urls


def validate_sources(cache):
    """An old/offline cache cannot masquerade as the current server release."""
    manifest = json.loads((cache / "probe-sources.json").read_text())
    expected = source_urls()
    if not isinstance(manifest, dict) or set(manifest) != set(expected):
        raise ValueError("source cache inventory differs from the selected release")
    for name, url in expected.items():
        entry = manifest[name]
        if not isinstance(entry, dict) or entry.get("url") != url:
            raise ValueError(f"source cache revision mismatch: {name}")
        content = (cache / name).read_bytes()
        if (entry.get("bytes") != len(content)
                or entry.get("sha256") != hashlib.sha256(content).hexdigest()):
            raise ValueError(f"source cache checksum mismatch: {name}")


def fetch(cache):
    urls = source_urls()
    cache.mkdir(parents=True, exist_ok=True)

    def download(item):
        name, url = item
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    content = response.read(32 * 1024 * 1024 + 1)
                    length = response.headers.get("Content-Length")
                if len(content) > 32 * 1024 * 1024 or (
                    length and len(content) != int(length)
                ):
                    raise ValueError(f"Unexpected/truncated artifact size: {url}")
                (cache / name).write_bytes(content)
                return name, {
                    "url": url,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    with ThreadPoolExecutor(max_workers=5) as pool:
        manifest = dict(pool.map(download, urls.items()))
    (cache / "probe-sources.json").write_text(json.dumps(manifest, indent=2))


def load_source(module_name, path, names=None, extra=None, imports=False):
    """Execute original AST nodes, not a rewritten/synthetic parsing algorithm."""
    import pydantic
    import regex
    from openai.types.responses import FunctionTool, NamespaceTool

    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    sys.modules[module_name] = module
    module.__dict__.update(
        {key: value for key, value in vars(typing).items() if not key.startswith("__")}
    )
    module.__dict__.update(
        {
            "json": json,
            "math": math,
            "functools": functools,
            "re": regex,
            "cached_property": functools.cached_property,
            "dataclass": dataclass,
            "field": field,
            "abstractmethod": abstractmethod,
            "logger": logging.getLogger(module_name),
            "TYPE_CHECKING": False,
            "FunctionTool": FunctionTool,
            "NamespaceTool": NamespaceTool,
        }
    )
    for name in (
        "BaseModel",
        "ConfigDict",
        "Field",
        "model_validator",
        "model_serializer",
    ):
        module.__dict__[name] = getattr(pydantic, name)
    module.__dict__.update(
        {key: value for key, value in (extra or {}).items() if not key.startswith("__")}
    )
    tree = ast.parse(path.read_text())
    if names is not None:
        tree.body = [
            node
            for node in tree.body
            if getattr(node, "name", None) in names
            or isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id in names
                for target in getattr(node, "targets", [getattr(node, "target", None)])
            )
        ]
    elif not imports:
        tree.body = [
            node
            for node in tree.body
            if not isinstance(node, (ast.Import, ast.ImportFrom))
        ]
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module == "__future__")
    ]
    tree.body.insert(
        0,
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    return module


def parser_class(cache):
    # Namespace packages bypass vllm/__init__ and CUDA imports. Protocol classes,
    # parser state, lexer, coercion and output serialization remain upstream code.
    for name in (
        "vllm",
        "vllm.parser",
        "vllm.parser.engine",
        "vllm.entrypoints",
        "vllm.entrypoints.serve",
        "vllm.entrypoints.serve.engine",
        "vllm.entrypoints.generate",
        "vllm.entrypoints.generate.base",
        "vllm.entrypoints.openai",
        "vllm.entrypoints.openai.chat_completion",
        "vllm.tool_parsers",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    chat = load_source(
        "vllm.entrypoints.chat_utils",
        cache / "chat_utils.py",
        {"make_tool_call_id", "get_tool_call_id_type", "_KIMI_MODEL_TYPES"},
        {"random_uuid": lambda: uuid.uuid4().hex},
    )
    base_protocol = load_source(
        "vllm.entrypoints.serve.engine.protocol",
        cache / "protocol.py",
        {"OpenAIBaseModel"},
    )
    protocol = load_source(
        "vllm.entrypoints.generate.base.protocol",
        cache / "generate_protocol.py",
        {
            "FunctionDefinition",
            "FunctionCall",
            "ToolCall",
            "DeltaFunctionCall",
            "DeltaToolCall",
            "ExtractedToolCallInformation",
            "DeltaMessage",
        },
        {"make_tool_call_id": chat.make_tool_call_id,
         "OpenAIBaseModel": base_protocol.OpenAIBaseModel},
    )
    chat_protocol = load_source(
        "vllm.entrypoints.openai.chat_completion.protocol",
        cache / "chat_protocol.py",
        {"ChatCompletionToolsParam"},
        vars(protocol),
    )
    load_source(
        "vllm.tool_parsers.utils",
        cache / "utils.py",
        {
            "_is_function_tool",
            "_extract_tool_info",
            "find_tool_name",
            "find_tool_properties",
            "iter_response_function_tool_info",
            "flat_namespace_tool_name",
            "_NAMESPACE_TOOL_SEPARATOR",
            "_is_json_finite",
            "extract_types_from_schema",
            "coerce_to_schema_type",
            "_TYPE_ALIASES",
        },
        {"ChatCompletionToolsParam": chat_protocol.ChatCompletionToolsParam},
    )
    load_source(
        "vllm.parser.abstract_parser",
        cache / "abstract_parser.py",
        {"StreamState", "Parser"},
        {"get_tool_call_id_type": chat.get_tool_call_id_type},
    )
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = logging.getLogger
    sys.modules[logger.__name__] = logger
    for name in (
        "events",
        "incremental_lexer",
        "token_id_scanner",
        "parser_engine_config",
        "streaming_parser_engine",
        "parser_engine",
    ):
        load_source("vllm.parser.engine." + name, cache / (name + ".py"), imports=True)
    qwen = load_source("vllm.parser.qwen3", cache / "qwen3.py", imports=True)
    # Ensure the selected CLI parser is in THIS release's registry.
    registry = load_source(
        "vllm._probe_tool_registry",
        cache / "tool_init.py",
        {"_TOOL_PARSERS_TO_REGISTER"},
    )
    assert registry._TOOL_PARSERS_TO_REGISTER["qwen3_coder"] == (
        "qwen3_engine_tool_parser",
        "Qwen3EngineToolParser",
    )
    return qwen.Qwen3Parser, chat_protocol.ChatCompletionToolsParam


def probe(cache):
    validate_sources(cache)
    import transformers
    from transformers import AutoTokenizer

    assert transformers.__version__ == "5.17.0", transformers.__version__
    tokenizer_dir = cache / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    for path in cache.glob("hf_*"):
        (tokenizer_dir / path.name.removeprefix("hf_")).write_bytes(path.read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    Parser, Tool = parser_class(cache)
    tool_dict = {
        "type": "function",
        "function": {
            "name": "inspect_scene",
            "description": "Read-only inspection",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "count": {"type": "integer"},
                    "enabled": {"type": "boolean"},
                    "point": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["label", "count", "enabled", "point"],
            },
        },
    }
    arguments = {
        "label": "cubo azul",
        "count": 2,
        "enabled": True,
        "point": [0.1, 0.2, 0.3],
    }
    tool = Tool(**tool_dict)
    request = types.SimpleNamespace(
        tools=[tool], tool_choice="auto", model="cosmos3-edge", include_reasoning=True
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Inspect the blue cube."}],
        tools=[tool_dict],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert prompt.endswith("<think>\n")
    history = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Inspect the blue cube."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "inspect_scene", "arguments": arguments},
                    }
                ],
            },
            {"role": "tool", "content": '{"ok":true}'},
        ],
        tools=[tool_dict],
        tokenize=False,
        add_generation_prompt=False,
    )
    xml = history[
        history.rindex("<tool_call>") : history.rindex("</tool_call>")
        + len("</tool_call>")
    ]
    sample = "<think>Read-only inspection.</think>" + xml
    parsed = Parser(tokenizer, tools=[tool]).extract_tool_calls(sample, request)
    assert parsed.tools_called and len(parsed.tool_calls) == 1, parsed
    call = parsed.tool_calls[0]
    assert call.type == "function" and call.id and call.function.name == "inspect_scene"
    assert json.loads(call.function.arguments) == arguments
    assert "<tool_call>" not in (parsed.content or "")
    no_thinking = Parser(
        tokenizer, tools=[tool], chat_template_kwargs={"enable_thinking": False}
    ).extract_tool_calls(xml, request)
    assert (
        no_thinking.tools_called
        and json.loads(no_thinking.tool_calls[0].function.arguments) == arguments
    )
    cases = 0
    token_ids = tokenizer.encode(sample, add_special_tokens=False)
    for chunk_size in (1, 2, 7, len(token_ids)):
        parser = Parser(tokenizer, tools=[tool])
        output = []
        for start in range(0, len(token_ids), chunk_size):
            ids = token_ids[start : start + chunk_size]
            delta = parser.parse_delta(
                tokenizer.decode(ids, skip_special_tokens=False),
                ids,
                request,
                finished=start + chunk_size >= len(token_ids),
            )
            if delta is not None:
                output.extend(delta.tool_calls)
        names = [c.function.name for c in output if c.function and c.function.name]
        combined = "".join(c.function.arguments or "" for c in output if c.function)
        assert names == ["inspect_scene"], names
        assert json.loads(combined) == arguments, combined
        cases += 1
    return {
        "evidence": "CPU upstream parser + official template/tokenizer; NOT generated inference or HTTP",
        "vllm_source_commit": VLLM_SHA,
        "model_revision": MODEL_SHA,
        "tokens": {
            s: tokenizer.encode(s, add_special_tokens=False)
            for s in ("<think>", "</think>", "<tool_call>", "</tool_call>")
        },
        "non_streaming": parsed.model_dump(),
        "thinking_disabled_fixture_passed": True,
        "streaming_chunk_sizes_passed": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--offline", action="store_true", help="Reuse captured source artifacts"
    )
    args = parser.parse_args()
    if not args.offline:
        fetch(args.cache)
    print(json.dumps(probe(args.cache), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
