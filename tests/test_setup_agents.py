"""Generators for the multi-platform MCP registration (setup_agents.py)."""

import json
import sys
from pathlib import Path

import pytest
from conftest import REPO

sys.path.insert(0, str(REPO / "scripts"))
import setup_agents  # noqa: E402
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


# ── Codex: base config vs a `-p NAME` profile layer ──────────────────────


def test_codex_config_path_base_and_profile_layer(tmp_path):
    """A robot tool server should not be loaded into every coding session:
    `--codex-profile robot` targets $CODEX_HOME/robot.config.toml, which Codex
    layers only under `codex -p robot`."""
    home = tmp_path / "codex-home"
    assert setup_agents.codex_config_path(None, codex_home=home) == home / "config.toml"
    assert setup_agents.codex_config_path("robot", codex_home=home) == home / "robot.config.toml"
    for bad in ("", "../x", "a/b", "ro bot", "a.b"):
        with pytest.raises(ValueError):
            setup_agents.codex_config_path(bad, codex_home=home)


def test_codex_config_path_honours_codex_home_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "elsewhere"))
    assert setup_agents.codex_config_path(None) == tmp_path / "elsewhere" / "config.toml"
    monkeypatch.delenv("CODEX_HOME")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    assert setup_agents.codex_config_path(None) == tmp_path / "home" / ".codex" / "config.toml"


def test_codex_profile_write_leaves_the_base_config_alone(tmp_path, monkeypatch, capsys):
    """Driving setup_agents.main() with --write against a PRIVATE $CODEX_HOME: the block
    lands in the profile layer, the base config.toml is untouched, and the
    upsert is idempotent there too."""
    home = tmp_path / "codex-home"
    home.mkdir()
    base = home / "config.toml"
    base.write_text('model = "gpt-5.5"\n')
    monkeypatch.setenv("CODEX_HOME", str(home))
    # belt and braces: even a path bug must not reach the real ~/.codex
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "fake-home"))
    argv = ["setup_agents.py", "--host", "codex", "--codex-profile", "robot",
            "--camera", "l515", "--arm", "mock", "--python", PY, "--write"]
    monkeypatch.setattr(sys, "argv", argv)
    assert setup_agents.main() == 0
    assert setup_agents.main() == 0  # idempotent
    layer = home / "robot.config.toml"
    assert layer.exists()
    import tomllib

    data = tomllib.loads(layer.read_text())
    assert data["mcp_servers"]["cascade"]["command"] == PY
    assert data["mcp_servers"]["cascade"]["env"]["CASCADE_CAMERAS"] == "l515"
    assert layer.read_text().count("[mcp_servers.cascade]") == 1
    assert base.read_text() == 'model = "gpt-5.5"\n', "the base config must not change"
    out = capsys.readouterr().out
    assert "codex -p robot" in out, "tell the operator how the layer is loaded"
    assert not (tmp_path / "fake-home").exists()


def test_codex_default_path_is_unchanged_without_a_profile(tmp_path, monkeypatch, capsys):
    home = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "fake-home"))
    monkeypatch.setattr(sys, "argv", ["setup_agents.py", "--host", "codex", "--python", PY])
    assert setup_agents.main() == 0
    out = capsys.readouterr().out
    assert str(home / "config.toml") in out
    assert "codex -p" not in out
    assert not (home / "config.toml").exists(), "no --write, no file"
