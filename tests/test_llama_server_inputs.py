"""What reaches llama-server besides its argv: both llama.cpp brain scripts and the Brev brain (#410, #415 reviews).

Minor 1: llama-server v0.7.0 applies /etc/llama.cpp/config.ini, then ${XDG_CONFIG_HOME:-$HOME/.config}/llama.cpp/
config.ini, before the environment and argv, and `--help` too (common/arg.cpp `common_params_apply_system_config`, run
by every parse). Each takes ANY server option, no allowlist (common/preset.cpp): `tools = all` serves exec_shell_command
at /tools, `api-prefix = /v1` moves every route inside the OpenShell sandbox's /v1/**, `log-file = <path>` truncates
that file. So every llama-server run, the `--help` capability probe included, gets XDG_CONFIG_HOME=/dev/null (no
llama.cpp/config.ini can exist under it; arg.cpp:738 is its only reader), and a system file refuses the start before
the first run, unless the operator sets CASCADE_LLAMA_CONFIG_INI=1.
Minor 2: the env scrub lists names with `compgen -e`; a bash without it must refuse, not serve unscrubbed.
Minor 5: deploy/runtime/brain.py drops the same names as the scripts (one list, pinned here), and its server reads no
config.ini either.

Fakes only: the fake llama-server applies config.ini as llama.cpp does, logs every call (argv, config home, the files
it applied) and prints its argv and environment; each copy of a script names a private file in place of
/etc/llama.cpp/config.ini. Nothing is downloaded, built or served.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import shutil
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
#: llama-server's config home on every run: a path where llama.cpp/config.ini can never exist (ENOTDIR).
NO_CONFIG_HOME = "/dev/null"


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

    def server_calls(self) -> list[dict]:
        """Every llama-server run of the last script runs, in order: argv, config home, config files applied."""
        log = Path(self.env["LLAMA_SERVER"]).parent / "calls.jsonl"
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    def helper_runs(self) -> list[list[str]]:
        log = self.home / "warm.jsonl"
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def _fake_server(path: Path, help_text: str, system: Path) -> None:
    """llama-server v0.7.0 as far as config.ini goes: before anything else, `--help` included, it applies `system` (the
    copy's /etc/llama.cpp/config.ini), then ${XDG_CONFIG_HOME:-$HOME/.config}/llama.cpp/config.ini, whichever exist
    (std::filesystem::exists: false under /dev/null), and opens a `log-file` for writing (log.cpp: fopen "w"
    truncates it). Every call is logged beside it."""
    path.parent.mkdir(parents=True)
    path.write_text(f"#!{sys.executable}\n" + f"""import json, os, sys
home = os.environ.get('XDG_CONFIG_HOME') or os.path.join(os.environ['HOME'], '.config')
applied = [p for p in ({str(system)!r}, os.path.join(home, 'llama.cpp', 'config.ini')) if os.path.exists(p)]
for p in applied:
    for line in open(p):
        key, _, value = line.partition('=')
        if key.strip() == 'log-file':
            open(value.strip(), 'w').close()
with open({str(path.parent / "calls.jsonl")!r}, 'a') as log:
    log.write(json.dumps({{'argv': sys.argv[1:], 'XDG_CONFIG_HOME': os.environ.get('XDG_CONFIG_HOME'),
                          'applied': applied}}) + '\\n')
if sys.argv[1:] == ['--help']:
    print({help_text!r})
else:
    print(json.dumps({{'server_args': sys.argv[1:], 'env': dict(os.environ)}}))
""")
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
    _fake_server(server, "--mmproj --alias --chat-template-kwargs --reasoning", system)
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
    _fake_server(server, BOLT_HELP, system)
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


def _write_config(config_home: Path, text: str = HOSTILE_CONFIG_INI) -> Path:
    path = config_home / "llama.cpp" / "config.ini"
    path.parent.mkdir(parents=True)
    path.write_text(text)
    return path


def _victim(tmp_path: Path) -> Path:
    """A file a stray `log-file = <path>` line would truncate."""
    victim = tmp_path / "victim.log"
    victim.write_text("must survive\n")
    return victim


# ── Minor 1: config.ini ────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", [(), ("--check",)], ids=["serve", "check"])
@pytest.mark.parametrize("via", ["XDG_CONFIG_HOME", "HOME"])
def test_no_llama_server_run_reads_an_inherited_config_ini(brain, tmp_path, via, mode):
    """The caller's config home (an inherited XDG_CONFIG_HOME, or HOME/.config without one) holds a config.ini with
    `tools = all`, `api-prefix = /v1` and a stray `log-file`: no llama-server run applies it, the `--help` capability
    probe included (#415 review, Minor 1: --help parses fully, and `log-file` truncated that file)."""
    caller = tmp_path / "caller-config" if via == "XDG_CONFIG_HOME" else brain.home / ".config"
    victim = _victim(tmp_path)
    hostile = _write_config(caller, HOSTILE_CONFIG_INI + f"log-file = {victim}\n")
    result = brain.run(*mode, **({"XDG_CONFIG_HOME": str(caller)} if via == "XDG_CONFIG_HOME" else {}))
    assert result.returncode == 0, result.stdout + result.stderr
    calls = brain.server_calls()
    assert [c["argv"][:1] for c in calls] == [["--help"]] + ([] if mode else [["--model"]])
    assert [(c["XDG_CONFIG_HOME"], c["applied"]) for c in calls] == [(NO_CONFIG_HOME, [])] * len(calls)
    assert victim.read_text() == "must survive\n"
    assert hostile.read_text().startswith(HOSTILE_CONFIG_INI)          # the caller's own file stays as it was
    if not mode:
        server = started(result)
        assert server["env"]["XDG_CONFIG_HOME"] == NO_CONFIG_HOME
        assert llama_config_files(server["env"], brain.system) == []
        # Bolt's warm start still runs beside it (only behind a key), with the caller's config home.
        assert bool(brain.helper_runs()) == ("--slot-save-path" in server["server_args"]) == ("API_KEY_FILE" in brain.env)


@pytest.mark.parametrize("state", ["unwritable", "unset"])
def test_no_state_directory_is_needed_for_the_config_home(brain, tmp_path, state):
    """#415 review, Minor 2: the private config home under ${XDG_STATE_HOME:-~/.local/state} made Qwen on llama.cpp
    need a writable state directory, and its ancestors could be swapped. /dev/null needs nothing to exist."""
    if state == "unwritable":
        locked = tmp_path / "locked"
        locked.mkdir(mode=0o500)
        env = {"XDG_STATE_HOME": str(locked / "state")}
    else:
        env = {"XDG_STATE_HOME": ""}
    server = started(brain.run(**env))
    assert server["env"]["XDG_CONFIG_HOME"] == NO_CONFIG_HOME
    assert not (brain.home / ".local/state/cascade/llama-server-config").exists()


@pytest.mark.parametrize("mode", [(), ("--check",), ("--setup-only",)], ids=["serve", "check", "setup"])
@pytest.mark.parametrize("kind", ["file", "dangling symlink"])
def test_a_system_config_ini_refuses_before_any_llama_server_run(brain, tmp_path, kind, mode):
    """/etc/llama.cpp/config.ini applies to every run, `--help` too, and no flag turns it off: refuse before the first
    run (#415 review, Minor 1: the --help probe ran first and applied it), and say how to opt in."""
    victim = _victim(tmp_path)
    brain.system.parent.mkdir(parents=True)
    if kind == "file":
        brain.system.write_text(HOSTILE_CONFIG_INI + f"log-file = {victim}\n")
    else:                                    # read the moment its target appears
        brain.system.symlink_to(tmp_path / "not-yet")
    why = refused(brain.run(*mode))
    assert str(brain.system) in why and "CASCADE_LLAMA_CONFIG_INI=1" in why, why
    assert brain.server_calls() == []
    assert victim.read_text() == "must survive\n"
    assert brain.helper_runs() == []


def test_the_opt_in_serves_with_both_config_files_and_says_so(brain, tmp_path):
    brain.system.parent.mkdir(parents=True)
    brain.system.write_text("metrics = true\n")
    user = _write_config(tmp_path / "caller-config")
    result = brain.run(CASCADE_LLAMA_CONFIG_INI="1", XDG_CONFIG_HOME=str(tmp_path / "caller-config"))
    server = started(result)
    assert llama_config_files(server["env"], brain.system) == [brain.system, user]
    assert [c["applied"] for c in brain.server_calls()] == [[str(brain.system), str(user)]] * 2
    assert "WARNING" in result.stderr and "CASCADE_LLAMA_CONFIG_INI=1" in result.stderr, result.stderr
    assert str(brain.system) in result.stderr and str(user) in result.stderr


def test_the_opt_in_off_is_the_default(brain):
    off = started(brain.run(CASCADE_LLAMA_CONFIG_INI="0"))
    assert off["env"].pop("CASCADE_LLAMA_CONFIG_INI") == "0"
    assert off == started(brain.run())


@pytest.mark.parametrize("value", ["yes", "2", "1 ", "true"])
def test_a_bad_opt_in_value_is_refused(brain, value):
    assert "CASCADE_LLAMA_CONFIG_INI" in refused(brain.run(CASCADE_LLAMA_CONFIG_INI=value), code=2)
    assert brain.server_calls() == []


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
    route to every endpoint) and LLAMA_SERVER_SLOTS_DEBUG (slot prompt text in GET /slots) reached its server, and it
    read the deployment-wide config home's llama.cpp/config.ini."""
    brain = _brev_brain()
    hostile = {**HOSTILE_LLAMA_ENV, "LLAMA_SERVER_SLOTS_DEBUG": "1"}
    for name, value in {**hostile, **BENIGN_ENV}.items():
        monkeypatch.setenv(name, value)
    # The deployment-wide config home (deploy/runtime/runtime.py: $PAAI_DATA_DIR/config), shared with OpenClaw and
    # every other worker of the container uid, holding a llama.cpp config.ini (#415 review, Minor 3).
    shared = tmp_path / "paai/config"
    _write_config(shared)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(shared))
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
    assert env["XDG_CONFIG_HOME"] == NO_CONFIG_HOME
    assert llama_config_files(env, Path("/etc/llama.cpp/config.ini")) == []
