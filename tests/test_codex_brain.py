"""GPT-6-Astra as the brain THROUGH the Codex CLI (`codex exec`).

The user's Codex subscription has no OPENAI_API_KEY: the only way to reach the
model is the CLI itself. So this backend is a subprocess, and what it must get
right is the command line (a brain whose Codex session loads the user's own
MCP servers could drive the robot around CASCADE's harness), the one-prompt
rendering of the conversation, and honest failures (stderr in the error, a
killed process on timeout). All of that is checked against a FAKE `codex`
binary that records what it was given; one live smoke test runs the real one
when it is installed and logged in.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path

import pytest

from cascade.agent.llm import ToolCall
from cascade.config import load_profile

llm_mod = importlib.import_module("cascade.agent.llm")

# A JPEG is just bytes to this client; distinct payloads let the test tell
# which observation reached the CLI.
OLD_JPEG = b"\xff\xd8OLD-OBSERVATION\xff\xd9"
NEW_JPEG_A = b"\xff\xd8NEW-A-MEMORY-FRAME\xff\xd9"
NEW_JPEG_B = b"\xff\xd8NEW-B-CURRENT-VIEW\xff\xd9"

TOOLS = [
    {
        "name": "grasp_object",
        "description": "Grasp the named object.",
        "parameters": {
            "type": "object",
            "properties": {"label": {"type": "string", "description": "detector label"}},
            "required": ["label"],
        },
    },
    {
        "name": "get_observation",
        "description": "Look at the scene.",
        "parameters": {"type": "object", "properties": {}},
    },
]

# The fake records argv, the prompt it read from stdin, copies of every `-i`
# file and of the `--output-schema` file, then writes a canned final message
# to the `-o` path. Behaviour is steered through FAKE_CODEX_* variables, which
# the child inherits from the test's environment.
FAKE_CODEX_SRC = r'''
import json, os, shutil, stat, sys, time
rec = os.environ["FAKE_CODEX_RECORD"]
os.makedirs(os.path.join(rec, "images"), exist_ok=True)
argv = sys.argv[1:]
json.dump(argv, open(os.path.join(rec, "argv.json"), "w"))
open(os.path.join(rec, "pid"), "w").write(str(os.getpid()))
stdin_text = sys.stdin.read() if argv and argv[-1] == "-" else ""
open(os.path.join(rec, "stdin.txt"), "w").write(stdin_text)
open(os.path.join(rec, "stdin_is_regular_file"), "w").write(
    str(stat.S_ISREG(os.fstat(0).st_mode)))
out_path = schema_path = cd = None
i = n_img = 0
while i < len(argv):
    a = argv[i]
    if a in ("-i", "--image"):
        n_img += 1
        shutil.copy(argv[i + 1], os.path.join(rec, "images", "%02d%s" % (n_img, os.path.splitext(argv[i + 1])[1])))
        i += 2
        continue
    if a in ("-o", "--output-last-message"):
        out_path = argv[i + 1]; i += 2; continue
    if a == "--output-schema":
        schema_path = argv[i + 1]; i += 2; continue
    if a in ("-C", "--cd"):
        cd = argv[i + 1]; i += 2; continue
    i += 1
if schema_path:
    shutil.copy(schema_path, os.path.join(rec, "schema.json"))
json.dump({"cd": cd, "cd_isdir": bool(cd and os.path.isdir(cd)),
           "cd_entries": (os.listdir(cd) if cd and os.path.isdir(cd) else None),
           "env_has_openai_key": "OPENAI_API_KEY" in os.environ},
          open(os.path.join(rec, "cd.json"), "w"))
sleep = float(os.environ.get("FAKE_CODEX_SLEEP", "0"))
if sleep:
    time.sleep(sleep)
sys.stderr.write(os.environ.get("FAKE_CODEX_STDERR", "OpenAI Codex v0.160.1\n--------\nmodel: fake\n"))
if out_path and not os.environ.get("FAKE_CODEX_NO_OUTPUT"):
    open(out_path, "w").write(os.environ.get("FAKE_CODEX_OUTPUT", '{"text": "", "tool_calls": []}'))
sys.exit(int(os.environ.get("FAKE_CODEX_EXIT", "0")))
'''


class FakeCodex:
    def __init__(self, bin_path: Path, record: Path):
        self.bin = bin_path
        self.record = record

    def argv(self) -> list[str]:
        return json.loads((self.record / "argv.json").read_text())

    def stdin(self) -> str:
        return (self.record / "stdin.txt").read_text()

    def images(self) -> list[bytes]:
        return [p.read_bytes() for p in sorted((self.record / "images").iterdir())]

    def image_names(self) -> list[str]:
        return [p.name for p in sorted((self.record / "images").iterdir())]

    def schema(self) -> dict:
        return json.loads((self.record / "schema.json").read_text())

    def cd(self) -> dict:
        return json.loads((self.record / "cd.json").read_text())

    def pid(self) -> int:
        return int((self.record / "pid").read_text())

    def flag(self, name: str) -> str:
        """Value following a flag in argv (first occurrence)."""
        argv = self.argv()
        return argv[argv.index(name) + 1]


def write_fake_auth(codex_home: Path, token: str | None = "test-token", api_key=None) -> Path:
    codex_home.mkdir(parents=True, exist_ok=True)
    tokens = {"access_token": token, "refresh_token": "r", "account_id": "a"} if token is not None else {}
    (codex_home / "auth.json").write_text(
        json.dumps({"auth_mode": "chatgpt", "OPENAI_API_KEY": api_key, "tokens": tokens})
    )
    return codex_home


@pytest.fixture
def fake_codex_home(tmp_path, monkeypatch) -> Path:
    """A private $CODEX_HOME with a logged-in auth.json. Never the real one."""
    home = write_fake_auth(tmp_path / "codex-home")
    monkeypatch.setenv("CODEX_HOME", str(home))
    return home


@pytest.fixture
def fake_codex(tmp_path, monkeypatch, fake_codex_home) -> FakeCodex:
    script = tmp_path / "fake_codex.py"
    script.write_text(FAKE_CODEX_SRC)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "codex"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    record = tmp_path / "record"
    monkeypatch.setenv("FAKE_CODEX_RECORD", str(record))
    monkeypatch.setenv("CASCADE_CODEX_BIN", str(wrapper))
    for var in ("FAKE_CODEX_OUTPUT", "FAKE_CODEX_EXIT", "FAKE_CODEX_STDERR", "FAKE_CODEX_SLEEP",
                "FAKE_CODEX_NO_OUTPUT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return FakeCodex(wrapper, record)


def _client(**kw):
    return llm_mod.CodexExecClient(**kw)


CONVERSATION = [
    {"role": "user", "content": "pick up the red cube", "images": [OLD_JPEG]},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [ToolCall("grasp_object", {"label": "red cube"}, id="call_0")],
    },
    {
        "role": "tool",
        "tool_call_id": "call_0",
        "name": "grasp_object",
        "content": '{"ok": true, "grasped": "red cube", "verdict": "confirmed"}',
    },
    {"role": "user", "content": "Memory frames, oldest first; the LAST image is the current view.",
     "images": [NEW_JPEG_A, NEW_JPEG_B]},
]


# ── the one prompt ────────────────────────────────────────────────────────


def test_prompt_carries_system_tools_and_the_whole_conversation(fake_codex):
    client = _client()
    client.chat("SYSTEM PERSONA TEXT", CONVERSATION, tools=TOOLS)
    prompt = fake_codex.stdin()
    assert "SYSTEM PERSONA TEXT" in prompt
    # tool specs: name, description and the parameter schema, in OpenAI function shape
    assert "grasp_object" in prompt and "Grasp the named object." in prompt
    assert '"label"' in prompt and "detector label" in prompt
    assert "get_observation" in prompt
    # the harness contract
    assert "at most ONE tool call" in prompt
    assert "cannot run commands or read files" in prompt
    # conversation in order: user text, the prior assistant call, the tool result verbatim
    assert "pick up the red cube" in prompt
    assert 'called grasp_object {"label": "red cube"}' in prompt
    assert '{"ok": true, "grasped": "red cube", "verdict": "confirmed"}' in prompt
    assert prompt.index("pick up the red cube") < prompt.index("called grasp_object") < prompt.index('"grasped"')
    # the prompt travels on stdin as a file, with `-` as the PROMPT argument
    assert fake_codex.argv()[-1] == "-"
    assert (fake_codex.record / "stdin_is_regular_file").read_text() == "True"


def test_only_the_latest_observation_images_are_attached(fake_codex):
    client = _client()
    client.chat("sys", CONVERSATION, tools=TOOLS)
    # the two images of the LAST image-carrying message, as JPEG files, in order
    assert fake_codex.images() == [NEW_JPEG_A, NEW_JPEG_B]
    assert all(n.endswith(".jpg") for n in fake_codex.image_names())
    argv = fake_codex.argv()
    assert argv.count("-i") == 2
    prompt = fake_codex.stdin()
    # the older image is replaced by a marker, not sent
    assert prompt.count("[image omitted]") == 1
    assert OLD_JPEG not in prompt.encode()
    assert "2 image" in prompt  # the model is told what is attached and where


def test_max_images_keeps_the_newest(fake_codex):
    client = _client(max_images=1)
    client.chat("sys", CONVERSATION, tools=TOOLS)
    assert fake_codex.images() == [NEW_JPEG_B], "the last image is the current view"


def test_text_only_profile_sends_no_images(fake_codex):
    client = _client(supports_vision=False)
    client.chat("sys", CONVERSATION, tools=TOOLS)
    assert fake_codex.images() == []
    assert "-i" not in fake_codex.argv()


# ── the command line ──────────────────────────────────────────────────────


def test_command_line_isolates_the_brain_session(fake_codex):
    client = _client(model="gpt-6-astra", reasoning_effort="medium")
    client.chat("sys", [{"role": "user", "content": "hi"}], tools=TOOLS)
    argv = fake_codex.argv()
    assert argv[0] == "exec"
    # the user's ~/.codex/config.toml (its MCP servers, hooks) must never reach
    # the brain: a session that can call the robot MCP server directly bypasses
    # CASCADE's harness
    assert "--ignore-user-config" in argv
    assert "--ephemeral" in argv
    assert "--skip-git-repo-check" in argv
    assert fake_codex.flag("-s") == "read-only"
    assert fake_codex.flag("-m") == "gpt-6-astra"
    i = argv.index("-c")
    assert argv[i + 1] == "model_reasoning_effort=medium"
    assert "--output-schema" in argv and "-o" in argv
    assert fake_codex.flag("--color") == "never"
    # the user's 660 skills would otherwise be injected until the context budget
    assert "skip_host_skill_discovery" in argv[argv.index("--enable") + 1]
    # a private, EMPTY working root: no repo AGENTS.md leaks into the prompt
    cd = fake_codex.cd()
    assert cd["cd_isdir"] and cd["cd_entries"] == []
    assert not cd["env_has_openai_key"], "no API key is involved anywhere"


def test_profile_overrides_reach_the_command_line(fake_codex):
    client = _client(
        model="gpt-6-astra-mini",
        reasoning_effort="low",
        codex_config=["model_verbosity=low", 'sandbox_permissions=[]'],
    )
    client.chat("sys", [{"role": "user", "content": "hi"}])
    argv = fake_codex.argv()
    assert fake_codex.flag("-m") == "gpt-6-astra-mini"
    cs = [argv[i + 1] for i, a in enumerate(argv) if a == "-c"]
    assert cs[0] == "model_reasoning_effort=low"
    assert "model_verbosity=low" in cs and "sandbox_permissions=[]" in cs


def test_output_schema_satisfies_openai_strict_mode(fake_codex):
    """The strict schema REJECTS free-form objects ('properties is required
    for object schemas'), so tool arguments travel as a JSON string."""
    _client().chat("sys", [{"role": "user", "content": "hi"}], tools=TOOLS)
    schema = fake_codex.schema()
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    assert sorted(schema["required"]) == ["text", "tool_calls"]
    assert schema["properties"]["text"] == {"type": "string"}
    item = schema["properties"]["tool_calls"]["items"]
    assert item["additionalProperties"] is False
    assert sorted(item["required"]) == ["arguments_json", "name"]
    assert item["properties"]["arguments_json"] == {"type": "string"}
    assert item["properties"]["name"] == {"type": "string"}


def test_env_binary_override_wins_and_a_missing_binary_fails_loudly(fake_codex, monkeypatch):
    # CASCADE_CODEX_BIN (set by the fixture) beats the profile's codex_bin
    client = _client(codex_bin="/nonexistent/codex")
    client.chat("sys", [{"role": "user", "content": "hi"}])
    assert fake_codex.argv()[0] == "exec"
    monkeypatch.delenv("CASCADE_CODEX_BIN")
    with pytest.raises(RuntimeError, match="CASCADE_CODEX_BIN"):
        _client(codex_bin="/nonexistent/codex")


# ── parsing the final message ─────────────────────────────────────────────


def test_final_message_becomes_a_tool_call_with_dict_arguments(fake_codex, monkeypatch):
    monkeypatch.setenv(
        "FAKE_CODEX_OUTPUT",
        json.dumps({"text": "Looking first.", "tool_calls": [
            {"name": "get_observation", "arguments_json": json.dumps({"detail": True, "n": 2})}]}),
    )
    client = _client()
    resp = client.chat("sys", [{"role": "user", "content": "hi"}], tools=TOOLS)
    assert resp.text == "Looking first."
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert tc.name == "get_observation"
    assert tc.arguments == {"detail": True, "n": 2}
    assert tc.id == "call_0"
    resp2 = client.chat("sys", [{"role": "user", "content": "again"}], tools=TOOLS)
    assert resp2.tool_calls[0].id == "call_1", "ids must be unique within a session"


def test_only_the_first_tool_call_is_kept(fake_codex, monkeypatch):
    monkeypatch.setenv(
        "FAKE_CODEX_OUTPUT",
        json.dumps({"text": "", "tool_calls": [
            {"name": "get_observation", "arguments_json": "{}"},
            {"name": "grasp_object", "arguments_json": json.dumps({"label": "x"})}]}),
    )
    resp = _client().chat("sys", [{"role": "user", "content": "hi"}], tools=TOOLS)
    assert [tc.name for tc in resp.tool_calls] == ["get_observation"]


def test_malformed_arguments_json_is_kept_raw_like_the_openai_client(fake_codex, monkeypatch):
    monkeypatch.setenv(
        "FAKE_CODEX_OUTPUT",
        json.dumps({"text": "", "tool_calls": [{"name": "grasp_object", "arguments_json": "{not json"}]}),
    )
    resp = _client().chat("sys", [{"role": "user", "content": "hi"}], tools=TOOLS)
    assert resp.tool_calls[0].arguments == {"_raw": "{not json"}


def test_text_answer_without_tool_call(fake_codex, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_OUTPUT", json.dumps({"text": "The table is empty.", "tool_calls": []}))
    resp = _client().chat("sys", [{"role": "user", "content": "what do you see?"}], tools=TOOLS)
    assert resp.text == "The table is empty." and resp.tool_calls == []


# ── honest failures ───────────────────────────────────────────────────────


def test_nonzero_exit_raises_with_the_stderr_tail(fake_codex, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_EXIT", "2")
    monkeypatch.setenv("FAKE_CODEX_STDERR", "ERROR: not logged in\nrun `codex login` first\n")
    with pytest.raises(RuntimeError) as ei:
        _client().chat("sys", [{"role": "user", "content": "hi"}])
    msg = str(ei.value)
    assert "exit" in msg and "2" in msg
    assert "not logged in" in msg and "codex login" in msg


def test_unparsable_final_message_raises(fake_codex, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_OUTPUT", "I would rather chat than emit JSON")
    with pytest.raises(RuntimeError, match="(?i)pars"):
        _client().chat("sys", [{"role": "user", "content": "hi"}])


def test_missing_final_message_raises(fake_codex, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_NO_OUTPUT", "1")
    with pytest.raises(RuntimeError):
        _client().chat("sys", [{"role": "user", "content": "hi"}])


def test_timeout_kills_the_process_and_raises(fake_codex, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_SLEEP", "30")
    client = _client(timeout_s=0.8)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="(?i)timed out"):
        client.chat("sys", [{"role": "user", "content": "hi"}])
    assert time.monotonic() - t0 < 10
    # the stuck codex must not linger: the whole process group is killed
    pid = fake_codex.pid()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        # a reaped-but-still-listed zombie counts as dead too
        try:
            state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
        except OSError:
            break
        if state == "Z":
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"fake codex pid {pid} still alive after the timeout")


def test_subprocess_argv_is_logged_at_info_for_operators(fake_codex, caplog):
    """An operator reading the run log must be able to see exactly which
    codex command a step ran (model, effort, isolation flags) without
    reproducing it."""
    import logging

    caplog.set_level(logging.INFO, logger="cascade.agent.llm")
    _client(model="gpt-6-astra", reasoning_effort="low").chat(
        "sys", [{"role": "user", "content": "hi"}], tools=TOOLS)
    argv_lines = [r.getMessage() for r in caplog.records
                  if r.levelno == logging.INFO and "argv" in r.getMessage()]
    assert argv_lines, [r.getMessage() for r in caplog.records]
    line = argv_lines[0]
    for needle in ("exec", "--ignore-user-config", "-m gpt-6-astra",
                   "model_reasoning_effort=low", "--output-schema"):
        assert needle in line, line


def test_close_removes_the_temp_files(fake_codex):
    client = _client()
    client.chat("sys", [{"role": "user", "content": "hi"}, ], tools=TOOLS)
    root = Path(client.temp_root)
    assert root.exists()
    # per-call files do not accumulate between steps
    assert list(root.iterdir()) == []
    client.close()
    assert not root.exists()
    client.close()  # idempotent


# ── auth detection + the profile ──────────────────────────────────────────


def test_codex_auth_present_reads_the_tokens(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    home = write_fake_auth(tmp_path / "h1")
    monkeypatch.setenv("CODEX_HOME", str(home))
    assert llm_mod.codex_auth_present() is True
    write_fake_auth(tmp_path / "h1", token="   ")
    assert llm_mod.codex_auth_present() is False, "a blank token is not a login"
    write_fake_auth(tmp_path / "h1", token=None)
    assert llm_mod.codex_auth_present() is False
    write_fake_auth(tmp_path / "h1", token=None, api_key="sk-in-auth-json")
    assert llm_mod.codex_auth_present() is True, "auth_mode apikey stores the key in auth.json"
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "nowhere"))
    assert llm_mod.codex_auth_present() is False
    (tmp_path / "h2").mkdir()
    (tmp_path / "h2" / "auth.json").write_text("not json")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "h2"))
    assert llm_mod.codex_auth_present() is False


def test_codex_auth_present_defaults_to_home_dot_codex(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert llm_mod.codex_auth_present() is False
    write_fake_auth(tmp_path / ".codex")
    assert llm_mod.codex_auth_present() is True


def test_construction_fails_loudly_when_codex_is_not_logged_in(fake_codex, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(fake_codex.record / "empty-home"))
    with pytest.raises(RuntimeError, match="codex login"):
        _client()


def test_codex_astra_profile_builds_this_client(fake_codex):
    cfg = load_profile("llm", "codex_astra")
    assert cfg.get("type") == "codex_exec"
    client = llm_mod.make_llm(cfg)
    assert isinstance(client, llm_mod.CodexExecClient)
    assert client.model == "gpt-6-astra"
    assert client.reasoning_effort == "medium"
    assert client.supports_vision is True
    assert client.timeout_s == 240
    assert client.max_images is None
    # nothing in the profile is a credential
    for key in ("api_key", "api_key_env", "base_url"):
        assert cfg.get(key) is None
    client.close()


# ── the real thing ────────────────────────────────────────────────────────

_REAL_CODEX = os.environ.get("CASCADE_CODEX_BIN") or shutil.which("codex")
_LOGGED_IN = bool(getattr(llm_mod, "codex_auth_present", lambda: False)())


@pytest.mark.skipif(
    not (_REAL_CODEX and _LOGGED_IN),
    reason="needs the Codex CLI on PATH (or CASCADE_CODEX_BIN) and a `codex login`",
)
def test_live_codex_exec_returns_a_tool_call():
    """One real `codex exec` round trip with gpt-6-astra: a tiny tool-choice
    prompt must come back as a ToolCall. Measures the step latency too."""
    client = llm_mod.make_llm(load_profile("llm", "codex_astra"))
    system = (
        "You are the brain of a robot arm. You act ONLY through the tools. "
        "Before any manipulation you must look at the scene."
    )
    tools = [
        {"name": "get_observation", "description": "Look at the scene (camera + detections).",
         "parameters": {"type": "object", "properties": {}}},
        {"name": "task_done", "description": "Finish the task.",
         "parameters": {"type": "object", "properties": {"success": {"type": "boolean"},
                                                          "summary": {"type": "string"}},
                        "required": ["success", "summary"]}},
    ]
    t0 = time.monotonic()
    try:
        resp = client.chat(system, [{"role": "user", "content": "Tidy the table."}], tools=tools)
    finally:
        client.close()
    dt = time.monotonic() - t0
    print(f"\nlive codex exec ({client.model}, {client.reasoning_effort}): {dt:.1f} s -> "
          f"text={resp.text!r} tool_calls={resp.tool_calls}")
    assert resp.tool_calls, f"expected a tool call, got text only: {resp.text!r}"
    assert resp.tool_calls[0].name == "get_observation"
    assert isinstance(resp.tool_calls[0].arguments, dict)
