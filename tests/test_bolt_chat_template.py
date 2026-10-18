"""scripts/chat_templates/: cascade's chat template for Nemotron 3.5 Bolt keeps OpenClaw's per-session text out of
the cached prefix.

OpenClaw splits its system text at a prompt-cache boundary (`<!-- openclaw:attempt:DYNAMIC -->`): before it the
stable instructions, after it the per-session part (date, output directives, the Runtime line with the session ids).
The GGUF's template renders the whole system text and THEN the 49 tool schemas, so every new chat differed ~2.4k
tokens into a ~13.6k-token prompt and llama.cpp re-prefilled it all. Cascade's template renders the per-session part
after the tool block; everything else must render exactly as the GGUF's template does (rendered here with jinja2,
the engine HF templates are written for; llama.cpp's engine is checked live in the evidence report).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

jinja2 = pytest.importorskip("jinja2")

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = REPO / "scripts/chat_templates"
GGUF = TEMPLATES / "nemotron-3.5-bolt.gguf.jinja"
CASCADE = TEMPLATES / "nemotron-3.5-bolt-openclaw.jinja"
BOUNDARY = "<!-- openclaw:attempt:DYNAMIC -->"

STABLE = ("<!-- openclaw:attempt:STABLE -->\nYou are a personal assistant running inside OpenClaw.\n## Tooling\n"
          "Tools policy-filtered.\n<!-- /openclaw:attempt:STABLE -->\n")


def dynamic(session: str) -> str:
    return (f"{BOUNDARY}\n## Temporal Context\nCurrent date: 2026-10-18\n## Runtime\nRuntime: agent=main | "
            f"session=agent:main:dashboard:{session} | sessionId={session} | model=cascade/nemotron-3.5-bolt\n"
            "Reasoning=off; hidden unless on/stream.\n<!-- /openclaw:attempt:DYNAMIC -->")


TOOLS = [{"type": "function", "function": {
    "name": "cascade__pick_and_place", "description": "Pick an object and place it.",
    "parameters": {"type": "object", "properties": {"object": {"type": "string", "description": "what"},
                                                    "destination": {"type": "string", "enum": ["green square", "box"]}},
                   "required": ["object"]}}},
    {"type": "function", "function": {"name": "cascade__describe_scene", "description": "Describe the table.",
                                      "parameters": {"type": "object", "properties": {}}}}]
IMAGE = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}


def render(path: Path, **kw) -> str:
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda v, **_: json.dumps(v, ensure_ascii=False)
    return env.from_string(path.read_text()).render(add_generation_prompt=True, enable_thinking=False, **kw)


def chat(system: str, *later) -> list[dict]:
    return [{"role": "system", "content": system},
            {"role": "user", "content": "[Sun 2026-10-18 08:16 UTC] What do you see on the table?"}, *later]


TOOL_TURNS = (
    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
        "name": "cascade__describe_scene", "arguments": {}}}]},
    {"role": "tool", "content": [IMAGE, {"type": "text", "text": "A green cube and a red can."}]},
    {"role": "assistant", "content": "A green cube and a red can."},
    {"role": "user", "content": "Put the green cube in the green square."})


def test_the_vendored_gguf_template_is_the_pinned_ggufs_own():
    # tokenizer.chat_template of NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf (sha256 91d63363..., pinned by
    # deploy/brev/profiles/nemotron-3.5-bolt-24gb.json), byte for byte.
    assert hashlib.sha256(GGUF.read_bytes()).hexdigest() == \
        "6521afc75afb0661e3558e7e6dee5f2d15de8be1c4dece8171935e778ea283f4"


@pytest.mark.parametrize("messages, tools", [
    # perception: one image + the prompt, no system text, no tools
    ([{"role": "user", "content": [IMAGE, {"type": "text", "text": "List every object."}]}], None),
    # a system text WITHOUT OpenClaw's boundary, with tools, over a multi-turn chat with tool calls
    (chat("You are a robot assistant.", *TOOL_TURNS), TOOLS),
    # OpenClaw's boundary but no tools (nothing to move the per-session text behind): unchanged
    (chat(STABLE + dynamic("1e6f762d"), *TOOL_TURNS), None),
    # no system message at all, with tools
    ([{"role": "user", "content": "hi"}], TOOLS),
])
def test_prompts_without_the_boundary_or_tools_render_exactly_as_the_gguf_template(messages, tools):
    kw = {"messages": copy.deepcopy(messages), **({"tools": tools} if tools else {})}
    assert render(CASCADE, **kw) == render(GGUF, **kw)


def test_the_per_session_text_moves_after_the_tool_block_and_nothing_else_changes():
    later = TOOL_TURNS
    got = render(CASCADE, messages=chat(STABLE + dynamic("1e6f762d"), *later), tools=TOOLS)
    # The GGUF template on the stable text alone, with the per-session text spliced in at the end of the system turn.
    plain = render(GGUF, messages=chat(STABLE.rstrip(), *later), tools=TOOLS)
    end = plain.index("</IMPORTANT>") + len("</IMPORTANT>")
    assert plain[end:].startswith("<|im_end|>\n<|im_start|>user\n")
    assert got == plain[:end] + "\n\n" + dynamic("1e6f762d") + plain[end:]
    # Inside the system turn, after every tool schema.
    system_turn = got[:got.index("<|im_end|>")]
    assert system_turn.index("cascade__describe_scene") < system_turn.index(BOUNDARY) < system_turn.index("sessionId=")


def test_two_chats_share_the_whole_system_text_and_tool_block():
    one = render(CASCADE, messages=chat(STABLE + dynamic("1e6f762d-0b80")), tools=TOOLS)
    two = render(CASCADE, messages=chat(STABLE + dynamic("92798453-b3e7")), tools=TOOLS)
    shared = next(i for i, (a, b) in enumerate(zip(one, two)) if a != b)
    assert one[:shared].rstrip().endswith("</IMPORTANT>\n\n" + BOUNDARY + "\n## Temporal Context\nCurrent date: "
                                          "2026-10-18\n## Runtime\nRuntime: agent=main | session=agent:main:dashboard:")
    # The GGUF's own template diverges before the tools: nothing after the session line can be reused.
    a = render(GGUF, messages=chat(STABLE + dynamic("1e6f762d-0b80")), tools=TOOLS)
    b = render(GGUF, messages=chat(STABLE + dynamic("92798453-b3e7")), tools=TOOLS)
    first = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
    assert first < a.index("# Tools")


def test_a_later_turn_of_the_same_chat_extends_the_first_prompt():
    # Same-chat prefix reuse is unchanged: each request's prompt starts with the previous one.
    first = render(CASCADE, messages=chat(STABLE + dynamic("1e6f762d")), tools=TOOLS)
    later = render(CASCADE, messages=chat(STABLE + dynamic("1e6f762d"), *TOOL_TURNS), tools=TOOLS)
    head = first[:first.rindex("<|im_start|>assistant\n")]
    assert later.startswith(head)


def test_a_repeated_boundary_keeps_all_its_text():
    text = STABLE + dynamic("x") + "\n" + BOUNDARY + "\nmore"
    got = render(CASCADE, messages=chat(text), tools=TOOLS)
    assert got.count(BOUNDARY) == 2 and "\nmore<|im_end|>" in got
    assert got.index("</IMPORTANT>") < got.index(BOUNDARY)
