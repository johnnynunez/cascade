"""What reaches llama-server besides its argv: both llama.cpp brain scripts and the Brev brain (#410 review).

Minor 1: llama-server v0.7.0 applies /etc/llama.cpp/config.ini, then ${XDG_CONFIG_HOME:-$HOME/.config}/llama.cpp/
config.ini, before the environment and argv (common/arg.cpp `common_params_apply_system_config`). Each takes ANY
server option, no allowlist (common/preset.cpp): `tools = all` serves exec_shell_command at /tools, `api-prefix = /v1`
moves every route inside the OpenShell sandbox's /v1/**. The user file follows the caller's XDG_CONFIG_HOME or HOME,
so the scripts start the server with an empty private config home, and a system file refuses the start, unless the
operator sets CASCADE_LLAMA_CONFIG_INI=1.
Minor 2: the env scrub lists names with `compgen -e`; a bash without it must refuse, not serve unscrubbed.
Minor 5: deploy/runtime/brain.py drops the same names as the scripts (one list, pinned here).

Fakes only: the fake llama-server prints its argv and environment, and each copy of a script names a private file in
place of /etc/llama.cpp/config.ini. Nothing is downloaded, built or served.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from inherited_server_env import (BENIGN_ENV, HOSTILE_CONFIG_INI, HOSTILE_LLAMA_ENV, dropped, leaked,
                                  llama_config_files)

REPO = Path(__file__).resolve().parents[1]
QWEN = "scripts/serve_qwen_llamacpp.sh"
BOLT = "scripts/serve_nemotron_bolt_llamacpp.sh"
BOLT_MANIFEST = "deploy/brev/profiles/nemotron-3.5-bolt-24gb.json"
BOLT_TEMPLATE = "scripts/chat_templates/nemotron-3.5-bolt-openclaw.jinja"
BOLT_HELP = ("--mmproj --alias --chat-template-kwargs --chat-template-file --reasoning --spec-draft-model "
             "--api-key-file --spec-type none,draft-simple,draft-dflash --load-mode MODE --slot-save-path PATH")
#: The one system file llama.cpp v0.7.0 reads outside Windows (common/arg.cpp:734); each script names it on this line.
SYSTEM_LINE = "LLAMA_SYSTEM_CONFIG=/etc/llama.cpp/config.ini"
#: Where the scripts keep llama-server's empty config home, under ${XDG_STATE_HOME:-$HOME/.local/state}.
CONFIG_HOME = "cascade/llama-server-config"


@dataclass
class Brain:
    script: Path
    env: dict
    home: Path
    system: Path   # what this copy of the script reads in place of /etc/llama.cpp/config.ini

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(self.script), *args], env={**self.env, **env},
                              capture_output=True, text=True, timeout=60)

    def run_without_compgen(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        """The script in a bash that lacks `compgen` (a minimal build without programmable completion)."""
        return subprocess.run(["bash", "-c", 'enable -n compgen && . "$0" "$@"', str(self.script), *args],
                              env={**self.env, **env}, capture_output=True, text=True, timeout=60)

    def own_config_home(self) -> Path:
        return self.home / ".local/state" / CONFIG_HOME

    def helper_runs(self) -> list[list[str]]:
        log = self.home / "warm.jsonl"
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def _fake_server(path: Path, help_text: str) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                    f"if sys.argv[1:] == ['--help']:\n    print({help_text!r})\n"
                    "else:\n    print(json.dumps({'server_args': sys.argv[1:], 'env': dict(os.environ)}))\n")
    path.chmod(0o755)


def _copy_script(repo: Path, rel: str, system: Path) -> Path:
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text((REPO / rel).read_text().replace(SYSTEM_LINE, f"LLAMA_SYSTEM_CONFIG={shlex.quote(str(system))}"))
    target.chmod(0o755)
    return target


def _qwen(repo: Path, system: Path) -> tuple[Path, dict]:
    script = _copy_script(repo, QWEN, system)
    models = repo / "models/qwen3.8-27b"
    models.mkdir(parents=True)
    for name in ("Qwen3.8-27B-UD-Q4_K_XL.gguf", "mmproj-BF16.gguf"):
        (models / name).write_bytes(b"GGUF" + b"\0" * 20)
    server = repo / "srv/bin/llama-server"
    _fake_server(server, "--mmproj --alias --chat-template-kwargs --reasoning")
    return script, {"LLAMA_SERVER": str(server), "CASCADE_QWEN_MODEL": str(models / "Qwen3.8-27B-UD-Q4_K_XL.gguf"),
                    "CASCADE_QWEN_MMPROJ": str(models / "mmproj-BF16.gguf")}


def _bolt(repo: Path, system: Path) -> tuple[Path, dict]:
    script = _copy_script(repo, BOLT, system)
    for rel in ("deploy/brev/fetch_models.py", BOLT_TEMPLATE):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, repo / rel)
    manifest = json.loads((REPO / BOLT_MANIFEST).read_text())
    models = repo / "models/nemotron-3.5-bolt"
    models.mkdir(parents=True)
    for item in manifest["quantization"]["files"]:     # the real names; small bodies
        body = b"GGUF" + item["filename"].encode()
        (models / item["filename"]).write_bytes(body)
        item.update(size_bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    licence = b"OpenMDW License Agreement, version 1.1\n"
    (models / "LICENSE").write_bytes(licence)
    manifest["source_model"]["license"].update(size_bytes=len(licence), sha256=hashlib.sha256(licence).hexdigest())
    (repo / BOLT_MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    (repo / BOLT_MANIFEST).write_text(json.dumps(manifest))
    (repo / "scripts/brain_warm_start.py").write_text(
        "import json, os, sys\nwith open(os.path.join(os.environ['HOME'], 'warm.jsonl'), 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n")
    server = repo / "srv/bin/llama-server"
    _fake_server(server, BOLT_HELP)
    return script, {"LLAMA_SERVER": str(server)}


@pytest.fixture(params=["qwen", "bolt", "bolt warm start"])
def brain(request, tmp_path) -> Brain:
    """One brain script with a clean environment: no XDG_* and no llama.cpp variables from the test host."""
    system = tmp_path / "etc/llama.cpp/config.ini"
    script, extra = (_qwen if request.param == "qwen" else _bolt)(tmp_path / "repo", system)
    home = tmp_path / "home"
    home.mkdir()
    env = {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(home), "PY": sys.executable,
           "CASCADE_LLAMA_CUDA_ARCH": "121", **extra}
    if request.param == "bolt warm start":
        key = tmp_path / "brain.key"
        key.write_text("config-ini-test-key\n")
        key.chmod(0o600)
        env["API_KEY_FILE"] = str(key)
    return Brain(script=script, env=env, home=home, system=system)


def started(result: subprocess.CompletedProcess) -> dict:
    """argv and environment of the (fake) llama-server the script exec'd."""
    assert result.returncode == 0, result.stdout + result.stderr
    return next(json.loads(line) for line in result.stdout.splitlines() if '"server_args"' in line)


def refused(result: subprocess.CompletedProcess, code: int = 1) -> str:
    assert result.returncode == code and '"server_args"' not in result.stdout, result.stdout + result.stderr
    return result.stderr


def _write_config(config_home: Path) -> Path:
    path = config_home / "llama.cpp" / "config.ini"
    path.parent.mkdir(parents=True)
    path.write_text(HOSTILE_CONFIG_INI)
    return path


# ── Minor 1: config.ini ────────────────────────────────────────────────────


@pytest.mark.parametrize("via", ["XDG_CONFIG_HOME", "HOME"])
def test_an_inherited_config_ini_never_configures_the_server(brain, tmp_path, via):
    """The caller's config home (an inherited XDG_CONFIG_HOME, or HOME/.config without one) holds a config.ini with
    `tools = all` and `api-prefix = /v1`: the server starts with an empty private config home instead."""
    caller = tmp_path / "caller-config" if via == "XDG_CONFIG_HOME" else brain.home / ".config"
    hostile = _write_config(caller)
    server = started(brain.run(**({"XDG_CONFIG_HOME": str(caller)} if via == "XDG_CONFIG_HOME" else {})))
    assert llama_config_files(server["env"], brain.system) == []
    own = brain.own_config_home()
    assert server["env"]["XDG_CONFIG_HOME"] == str(own)
    assert stat.S_IMODE(own.stat().st_mode) == 0o700 and list(own.iterdir()) == []
    assert hostile.read_text() == HOSTILE_CONFIG_INI           # the caller's own file stays as it was
    # Bolt's warm start still runs beside it (only behind a key).
    assert bool(brain.helper_runs()) == ("--slot-save-path" in server["server_args"]) == ("API_KEY_FILE" in brain.env)


def test_the_private_config_home_follows_the_xdg_state_home(brain, tmp_path):
    server = started(brain.run(XDG_STATE_HOME=str(tmp_path / "state")))
    assert server["env"]["XDG_CONFIG_HOME"] == str(tmp_path / "state" / CONFIG_HOME)


@pytest.mark.parametrize("kind", ["file", "dangling symlink"])
def test_a_system_config_ini_refuses_the_start(brain, tmp_path, kind):
    """/etc/llama.cpp/config.ini applies to every start and no flag turns it off: refuse, and say how to opt in."""
    brain.system.parent.mkdir(parents=True)
    if kind == "file":
        brain.system.write_text(HOSTILE_CONFIG_INI)
    else:                                    # read the moment its target appears
        brain.system.symlink_to(tmp_path / "not-yet")
    for args in ((), ("--check",)):
        why = refused(brain.run(*args))
        assert str(brain.system) in why and "CASCADE_LLAMA_CONFIG_INI=1" in why, why
    assert brain.helper_runs() == []


def test_the_opt_in_serves_with_both_config_files_and_says_so(brain, tmp_path):
    brain.system.parent.mkdir(parents=True)
    brain.system.write_text("metrics = true\n")
    user = _write_config(tmp_path / "caller-config")
    result = brain.run(CASCADE_LLAMA_CONFIG_INI="1", XDG_CONFIG_HOME=str(tmp_path / "caller-config"))
    server = started(result)
    assert llama_config_files(server["env"], brain.system) == [brain.system, user]
    assert "WARNING" in result.stderr and "CASCADE_LLAMA_CONFIG_INI=1" in result.stderr, result.stderr
    assert str(brain.system) in result.stderr and str(user) in result.stderr
    assert not brain.own_config_home().exists()


def test_the_opt_in_off_is_the_default(brain):
    off = started(brain.run(CASCADE_LLAMA_CONFIG_INI="0"))
    assert off["env"].pop("CASCADE_LLAMA_CONFIG_INI") == "0"
    assert off == started(brain.run())


@pytest.mark.parametrize("value", ["yes", "2", "1 ", "true"])
def test_a_bad_opt_in_value_is_refused(brain, value):
    assert "CASCADE_LLAMA_CONFIG_INI" in refused(brain.run(CASCADE_LLAMA_CONFIG_INI=value), code=2)


@pytest.mark.parametrize("problem", ["holds a config.ini", "holds anything", "a symlink", "a file"])
def test_a_config_home_that_cannot_be_private_and_empty_refuses_the_start(brain, tmp_path, problem):
    own = brain.own_config_home()
    own.parent.mkdir(parents=True)
    if problem == "holds a config.ini":
        own.mkdir(mode=0o700)
        _write_config(own)
    elif problem == "holds anything":
        own.mkdir(mode=0o700)
        (own / ".keep").write_text("")
    elif problem == "a symlink":
        (tmp_path / "elsewhere").mkdir()
        own.symlink_to(tmp_path / "elsewhere")
    else:
        own.write_text("a file where the directory goes")
    assert str(own) in refused(brain.run())
    assert brain.helper_runs() == []


def test_an_existing_empty_config_home_is_made_private(brain):
    own = brain.own_config_home()
    own.mkdir(parents=True)
    own.chmod(0o755)
    started(brain.run())
    assert stat.S_IMODE(own.stat().st_mode) == 0o700


def _shim(brain: Brain, tmp_path: Path, command: str, foreign: str) -> dict:
    """PATH with `command` answering for the config home as for a directory this user cannot use."""
    shims = tmp_path / "shims"
    shims.mkdir()
    (shims / command).write_text(f'#!/bin/bash\nif [[ "${{@: -1}}" == */llama-server-config ]]; then {foreign}; '
                                 f'else exec /usr/bin/{command} "$@"; fi\n')
    (shims / command).chmod(0o755)
    return {"PATH": f"{shims}{os.pathsep}{brain.env['PATH']}"}


@pytest.mark.parametrize("command, foreign", [("stat", "echo 4242"), ("ls", "exit 2")],
                         ids=["of another user", "cannot be listed"])
def test_a_config_home_that_cannot_be_checked_refuses_the_start(brain, tmp_path, command, foreign):
    # Another user's directory cannot be made here without root: `stat` answers another uid for it instead. An
    # `ls` that fails must not read as "empty".
    why = refused(brain.run(**_shim(brain, tmp_path, command, foreign)))
    assert str(brain.own_config_home()) in why
    assert brain.helper_runs() == []


def test_check_creates_no_config_home(brain):
    assert brain.run("--check").returncode == 0
    assert not brain.own_config_home().exists()


def test_both_scripts_name_the_system_config_llamacpp_reads():
    """llama.cpp v0.7.0 reads exactly one system file outside Windows; the tests replace this line in their copies."""
    for rel in (QWEN, BOLT):
        assert (REPO / rel).read_text().splitlines().count(SYSTEM_LINE) == 1, rel


# ── Minor 2: the scrub fails closed ────────────────────────────────────────


def test_a_bash_without_compgen_refuses_instead_of_serving_unscrubbed(brain):
    why = refused(brain.run_without_compgen(**HOSTILE_LLAMA_ENV))
    assert "compgen" in why


def test_the_scrub_runs_in_a_bash_with_compgen(brain):
    assert leaked(started(brain.run(**HOSTILE_LLAMA_ENV))["env"], HOSTILE_LLAMA_ENV) == {}


# ── Minor 5: one list for every llama-server cascade starts ────────────────


def _brev_brain():
    spec = importlib.util.spec_from_file_location("cascade_brev_brain", REPO / "deploy/runtime/brain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scrub_patterns(rel: str) -> list[str]:
    """The `case` patterns of a script's env scrub (`case "$name" in A|B) unset "$name" ;; esac`)."""
    (line,) = [x for x in (REPO / rel).read_text().splitlines() if x.strip().startswith('case "$name" in ')]
    return line.split(" in ", 1)[1].split(")", 1)[0].strip().split("|")


def test_the_scripts_and_the_brev_brain_drop_one_list():
    assert _scrub_patterns(QWEN) == _scrub_patterns(BOLT) == list(_brev_brain().LLAMA_SERVER_ENV)


def test_the_brev_brain_starts_llama_server_without_the_inherited_settings(monkeypatch, tmp_path):
    """deploy/runtime/brain.py dropped only LLAMA_ARG_*: LLAMA_API_KEY (an extra key), AIP_* (another port and a
    route to every endpoint) and LLAMA_SERVER_SLOTS_DEBUG (slot prompt text in GET /slots) reached its server."""
    brain = _brev_brain()
    hostile = {**HOSTILE_LLAMA_ENV, "LLAMA_SERVER_SLOTS_DEBUG": "1"}
    for name, value in {**hostile, **BENIGN_ENV}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(brain.signal, "signal", lambda *args: None)
    dump = tmp_path / "server-env.json"
    server = tmp_path / "fake_llama_server.py"
    server.write_text(f"import json, os, time\nopen({str(dump)!r}, 'w').write(json.dumps(dict(os.environ)))\n"
                      "time.sleep(.3)\n")
    plan = {"argv": [sys.executable, str(server)], "profile": "qwen", "unit": "synthetic-test-only",
            "plan_sha256": "synthetic", "state_file": str(tmp_path / "state.json")}
    assert brain.serve(plan) == 0
    env = json.loads(dump.read_text())
    assert leaked(env, hostile) == {}
    assert dropped(env) == {} and (env["CUDA_VISIBLE_DEVICES"], env["GGML_SCHED_DEBUG"]) == ("0", "1")
