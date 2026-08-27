"""Generators for the multi-platform MCP registration (setup_agents.py)."""

import json
import sys

from conftest import REPO

sys.path.insert(0, str(REPO / "scripts"))
from setup_agents import (  # noqa: E402
    claude_add_command,
    claude_mcp_json,
    codex_toml_block,
    codex_upsert,
    openclaw_command,
    openclaw_json_block,
    server_env,
)

ENV = server_env("l515", "mock", ":1")
PY = "/usr/bin/python3"


def test_codex_toml_block_shape():
    block = codex_toml_block(PY, ENV)
    assert block.startswith("[mcp_servers.cascade]\n")
    assert 'command = "/usr/bin/python3"' in block
    assert 'args = ["-m", "cascade.apps.mcp_server"]' in block
    assert "[mcp_servers.cascade.env]" in block
    assert 'CASCADE_CAMERAS = "l515"' in block
    # tomllib must parse it
    import tomllib

    data = tomllib.loads(block)
    assert data["mcp_servers"]["cascade"]["env"]["CASCADE_ARM"] == "mock"


def test_codex_upsert_preserves_and_replaces():
    existing = (
        'model = "gpt-5.5"\n\n'
        "[mcp_servers.other]\n"
        'command = "node"\n'
        'args = ["x.js"]\n'
    )
    merged = codex_upsert(existing, codex_toml_block(PY, ENV))
    assert 'model = "gpt-5.5"' in merged
    assert "[mcp_servers.other]" in merged
    assert "[mcp_servers.cascade]" in merged
    # idempotent: replacing again leaves exactly one entry
    env2 = server_env("mock", "mock", ":1")
    merged2 = codex_upsert(merged, codex_toml_block(PY, env2))
    assert merged2.count("[mcp_servers.cascade]") == 1
    assert 'CASCADE_CAMERAS = "mock"' in merged2 and 'CASCADE_CAMERAS = "l515"' not in merged2
    assert "[mcp_servers.other]" in merged2
    import tomllib

    data = tomllib.loads(merged2)
    assert set(data["mcp_servers"]) == {"other", "cascade"}


def test_claude_mcp_json_merges():
    existing = json.dumps({"mcpServers": {"other": {"command": "x"}}})
    out = json.loads(claude_mcp_json(existing, PY, ENV))
    assert set(out["mcpServers"]) == {"other", "cascade"}
    entry = out["mcpServers"]["cascade"]
    assert entry["type"] == "stdio"
    assert entry["args"] == ["-m", "cascade.apps.mcp_server"]
    assert entry["env"]["CASCADE_CAMERAS"] == "l515"


def test_claude_add_command_shape():
    cmd = claude_add_command(PY, ENV)
    assert cmd.startswith("claude mcp add --scope user ")
    assert "--env CASCADE_CAMERAS=l515" in cmd
    assert cmd.endswith("cascade -- /usr/bin/python3 -m cascade.apps.mcp_server")


def test_openclaw_outputs():
    cmd = openclaw_command(PY, ENV)
    # verified against OpenClaw 2026.7.1-2: the subcommand is `mcp add`
    # (probe-before-save), args are repeatable --arg flags, and --cwd keeps
    # YOLOE's CWD-relative text-encoder resolution working.
    assert cmd.startswith("openclaw mcp add cascade ")
    assert "--arg -m --arg cascade.apps.mcp_server" in cmd
    assert "--cwd" in cmd
    # OpenClaw blocks PYTHONPATH for stdio servers (startup safety) — emitting
    # it would only produce a scary warning; the venv needs the editable
    # install instead.
    assert "PYTHONPATH" not in cmd
    assert "--env CASCADE_CAMERAS=l515" in cmd
    data = json.loads(openclaw_json_block(PY, ENV))
    assert data["mcpServers"]["cascade"]["command"] == PY
