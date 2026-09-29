"""Run the real front door with private ownership receipts and a native CLI double."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from urllib.parse import quote

import pytest

from cascade.apps.process_owner import load_owner, register_process
from conftest import start_sleeping_process

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def installed(tmp_path):
    repo = tmp_path / "checkout with spaces"
    for name in ("run.sh", "scripts/dashboard.py", "src/cascade/apps/process_owner.py"):
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    state_root = tmp_path / "installation state"
    state = state_root / "profile-cascade-demo"
    owner = load_owner(state, repo, "cascade-demo", create=True)
    gateway = start_sleeping_process("fixture-gateway", start_new_session=True)
    register_process(state, owner, gateway.pid, "gateway_child")
    config = state / "openclaw/openclaw.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"gateway": {"port": 23456, "auth": {"token": "current/private+token"}}}))
    install = repo / "runs/.install/env.sh"
    install.parent.mkdir(parents=True)
    install.write_text("\n".join(f"export {key}={shlex.quote(value)}" for key, value in {
        "PY": sys.executable, "CASCADE_INSTALL_PROFILE": "spark", "CASCADE_OPENCLAW_PROFILE": "cascade-demo",
        "CASCADE_LAUNCH_STATE": str(state_root), "INSTALL_MARKER": "restored",
    }.items()) + "\n")
    cli = repo / ".openclaw-cli/bin/openclaw"
    cli.parent.mkdir(parents=True)
    cli.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
from urllib.parse import quote
args = sys.argv[1:]
keys = ("OPENCLAW_STATE_DIR", "OPENCLAW_CONFIG_PATH", "OPENCLAW_PROFILE", "NODE_COMPILE_CACHE", "INSTALL_MARKER")
with open(os.environ["CLI_LOG"], "a") as log:
    log.write(json.dumps({"args": args, "env": {key: os.environ.get(key) for key in keys}}) + "\\n")
assert args[:2] == ["--profile", "cascade-demo"]
assert "OPENCLAW_GATEWAY_TOKEN" not in os.environ
assert "OPENCLAW_GATEWAY_PORT" not in os.environ
if args[2:3] == ["health"]:
    print(os.environ.get("HEALTH_REPLY", '{"ok":true}'))
elif args[2:] == ["dashboard", "--json"]:
    config = json.loads(Path(os.environ["OPENCLAW_CONFIG_PATH"]).read_text())["gateway"]
    url = f'http://127.0.0.1:{config["port"]}/#token={quote(config["auth"]["token"], safe="")}'
    print(os.environ.get("DASHBOARD_REPLY", json.dumps({"ok": True, "url": url})))
elif args[2:] == ["dashboard"]:
    Path(os.environ["BROWSER_MARKER"]).write_text("opened")
else:
    raise SystemExit("Unexpected native CLI command")
''')
    cli.chmod(0o755)
    env = {**os.environ, "CLI_LOG": str(tmp_path / "cli.jsonl"), "BROWSER_MARKER": str(tmp_path / "browser"),
           "OPENCLAW_STATE_DIR": str(tmp_path / "personal"), "OPENCLAW_CONFIG_PATH": str(tmp_path / "personal.json"),
           "OPENCLAW_PROFILE": "personal", "OPENCLAW_GATEWAY_TOKEN": "personal-token",
           "OPENCLAW_GATEWAY_PORT": "1", "CASCADE_LAUNCH_STATE": str(tmp_path / "wrong-state")}
    try:
        yield repo, state, config, gateway, env
    finally:
        gateway.terminate()
        gateway.wait(timeout=5)


def invoke(installed, *arguments):
    repo, _, _, _, env = installed
    return subprocess.run(["bash", str(repo / "run.sh"), "dashboard", *arguments],
                          cwd=repo.parent, env=env, capture_output=True, text=True, timeout=15)


def test_no_open_restores_install_and_profile_and_reads_current_token_each_time(installed):
    repo, state, config, _, env = installed
    before = {p: p.read_bytes() for p in state.rglob("*") if p.is_file()}
    for token in ("current/private+token", "rotated-token?#&"):
        config.write_text(json.dumps({"gateway": {"port": 23456, "auth": {"token": token}}}))
        result = invoke(installed, "--no-open")
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "http://127.0.0.1:23456/#token=" + quote(token, safe="")
        assert not result.stderr
    calls = [json.loads(line) for line in Path(env["CLI_LOG"]).read_text().splitlines()]
    assert [call["args"][2] for call in calls] == ["health", "dashboard", "health", "dashboard"]
    assert calls[-1]["env"] == {
        "OPENCLAW_STATE_DIR": str(config.parent), "OPENCLAW_CONFIG_PATH": str(config),
        "OPENCLAW_PROFILE": "cascade-demo", "NODE_COMPILE_CACHE": str(config.parent / "cache/node-compile"),
        "INSTALL_MARKER": "restored",
    }
    assert not Path(env["BROWSER_MARKER"]).exists()
    assert not (repo / "runs/.launch").exists()
    assert {p: p.read_bytes() for p in state.rglob("*") if p.is_file() and p != config} == {
        p: contents for p, contents in before.items() if p != config}


def test_default_opens_native_dashboard(installed):
    result = invoke(installed)
    assert result.returncode == 0, result.stderr
    assert Path(installed[-1]["BROWSER_MARKER"]).read_text() == "opened"


@pytest.mark.parametrize("fault", ["stopped", "missing-owner", "foreign-owner", "missing-config", "missing-cli"])
def test_missing_or_wrong_stack_fails_without_opening_or_starting_anything(installed, fault):
    repo, state, config, gateway, env = installed
    if fault == "stopped":
        gateway.terminate()
        gateway.wait(timeout=5)
    elif fault == "missing-owner":
        (state / "owner.json").unlink()
    elif fault == "foreign-owner":
        owner = json.loads((state / "owner.json").read_text())
        owner["repo"] = str(repo.parent / "another checkout")
        (state / "owner.json").write_text(json.dumps(owner))
    elif fault == "missing-config":
        config.unlink()
    else:
        (repo / ".openclaw-cli/bin/openclaw").unlink()
    result = invoke(installed, "--no-open")
    assert result.returncode != 0
    assert "[dashboard] ERROR:" in result.stderr
    assert "not running" in result.stderr or "missing" in result.stderr or "start PAAI" in result.stderr
    assert not result.stdout
    assert not Path(env["CLI_LOG"]).exists()
    assert not Path(env["BROWSER_MARKER"]).exists()


@pytest.mark.parametrize("reply", ['{"ok":false}', '{"ok":1}', '[]', 'private-token-fixture'])
def test_unhealthy_gateway_fails_without_disclosing_diagnostics(installed, reply):
    env = installed[-1]
    env["HEALTH_REPLY"] = reply
    result = invoke(installed, "--no-open")
    assert result.returncode != 0
    assert "not running or healthy" in result.stderr
    assert "private-token-fixture" not in result.stdout + result.stderr
    assert not Path(env["BROWSER_MARKER"]).exists()
    assert len(Path(env["CLI_LOG"]).read_text().splitlines()) == 1


@pytest.mark.parametrize("reply", ['{"ok":false,"reason":"private-token-fixture"}', 'invalid private-token-fixture'])
def test_native_dashboard_failure_is_actionable_and_private(installed, reply):
    installed[-1]["DASHBOARD_REPLY"] = reply
    result = invoke(installed, "--no-open")
    assert result.returncode != 0
    assert "[dashboard] ERROR:" in result.stderr
    assert "private-token-fixture" not in result.stdout + result.stderr


def test_fresh_checkout_reports_stopped_stack_and_help_without_installing(tmp_path):
    repo = tmp_path / "fresh"
    for name in ("run.sh", "scripts/dashboard.py", "src/cascade/apps/process_owner.py"):
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    env = {"PATH": os.environ["PATH"]}
    result = subprocess.run(["bash", str(repo / "run.sh"), "dashboard", "--no-open"],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "stack is not running" in result.stderr
    help_text = subprocess.run(["bash", str(repo / "run.sh"), "--help"],
                               env=env, capture_output=True, text=True, timeout=10)
    assert "dashboard [--no-open]" in help_text.stdout
    assert not (repo / "runs").exists()
