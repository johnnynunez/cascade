"""B68: the legacy chat-host registrars forward the same runtime switches as launch.sh.

`scripts/hermes_demo.sh` (Hermes: register + test + chat, README's Hermes
row) and `scripts/openclaw_demo.sh` (OpenClaw, local-brain variant) are still
documented entry points: README's quick start and platform table,
install_hermes.sh's closing banner, `setup_agents.py --host hermes` and
docs/OPENCLAW_2.0_INTEGRATION_BRIEF.md point at them, and launch.sh (OpenClaw
only) has no Hermes path. Until B68 each registered a FIXED env (cameras, arm,
detector, offline flags, DISPLAY). A stdio MCP host starts the server with the
registered env, not the registering shell, so a runtime switch exported there
-- `CASCADE_GRASP_EXECUTOR=vla`, `CASCADE_VLA_PORT`,
`CASCADE_MCP_READONLY_LANE` -- silently did nothing: B63's defect, fixed then in
launch.sh and setup_agents.py only.

Both scripts now take the registry's rule (`cascade.apps.mcp_env`): their own
values first, every forwarded switch set in the shell after them, verbatim;
rig selectors and secrets never. Measured through the REAL scripts with only
the host CLIs doubled (`hermes`, `openclaw`, `ss`, `curl`, `uv`) and the
interpreter wrapped with `-S`, so the registry must come from the script's own
checkout, never from an installed cascade. Ports are from this item's block
(46900-46999); nothing here dials them.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO
from test_mcp_env_forwarding import _command_substitution_heredocs

SCRIPTS = REPO / "scripts"
HERMES, OPENCLAW = "hermes_demo.sh", "openclaw_demo.sh"
#: the flags every run passes: OpenClaw's --brain skip keeps the run off any model server
BASE_ARGS = {HERMES: [], OPENCLAW: ["--brain", "skip"]}
#: what each script writes itself (its flags and its own resolution), never replaced by the shell
OWN = {HERMES: ["PYTHONPATH", "CASCADE_CAMERAS", "CASCADE_ARM", "CASCADE_DETECTOR_MODEL", "DISPLAY",
                "YOLO_OFFLINE", "ULTRALYTICS_OFFLINE"],
       OPENCLAW: ["CASCADE_CAMERAS", "CASCADE_ARM", "CASCADE_DETECTOR_MODEL", "CASCADE_DETECT_CLASSES",
                  "YOLO_OFFLINE", "ULTRALYTICS_OFFLINE", "DISPLAY"]}
BLOCK = range(46900, 47000)
#: this item's sidecar ports; set in the shell of every run that builds a runtime from the entry
PORTS = {"CASCADE_GRASPGENX_PORT": "46901", "CASCADE_OCCUPANCY_PORT": "46902",
         "CASCADE_BRIDGE_PORT": "46903", "CASCADE_HUG_PORT": "46904"}
#: the B49 / B46 switches the item is about
SWITCHES = {"CASCADE_GRASP_EXECUTOR": "vla", "CASCADE_VLA_PORT": "46911", "CASCADE_MCP_READONLY_LANE": "1"}
#: read by the runtime, never copied from the registering shell (one per category)
NOT_FORWARDED_SAMPLE = {
    "CASCADE_ARMS": "so101_left,so101_right", "CASCADE_CAMERA": "mock_rgb", "CASCADE_ROBOT": "b68_robot",
    "CASCADE_BASE": "microduck_isaac", "CASCADE_RUN_DIR": "/b68/one-shared-run-dir", "CASCADE_LLM": "openai",
    "CASCADE_CODEX_BIN": "/b68/codex", "CASCADE_PREMOTION_CHECK": "0", "CASCADE_JUDGE": "fake",
    "CASCADE_JUDGE_TIMEOUT_S": "7", "CASCADE_MCP_SSE_KEEPALIVE_S": "3",
}
#: credentials a registering shell may carry; their VALUES must appear nowhere
SECRETS = {"CASCADE_MCP_TOKEN": "b68-mcp-bearer-5e0d", "CASCADE_MODEL_API_KEY": "b68-model-key-c41a",
           "HF_TOKEN": "hf_b68_secret_7f2b", "GITHUB_TOKEN": "ghp_b68_secret_0a9e",
           "OPENAI_API_KEY": "sk-b68-secret-31d4", "ANTHROPIC_API_KEY": "sk-ant-b68-secret-9c2e"}

_LOGGING_CLI = """import json, os, sys
with open(os.environ["HOST_LOG"], "a") as log:
    log.write(json.dumps([os.path.basename(sys.argv[0]), *sys.argv[1:]]) + "\\n")
sys.exit(int(os.environ.get("B68_EXIT_" + os.path.basename(sys.argv[0]).upper(), "0")))
"""
#: the interpreter the scripts call as "$PY". `-S`: no site-packages, so the
#: registry is importable only from the PYTHONPATH the script gives it (its own
#: checkout), never from whichever cascade the venv has installed.
_PYTHON = """import os, sys
args = sys.argv[1:]
if args == ["-c", "import clip"]:
    sys.exit(1)  # no CLIP fork: hermes_demo.sh stays on the closed-set detector (deterministic)
if args == ["-c", "import cascade"]:
    sys.exit(0)  # stands in for the editable install openclaw_demo.sh checks for
if os.environ.get("B68_REGISTRY_FAILS") and args[:2] == ["-m", "cascade.apps.mcp_env"]:
    sys.exit(3)  # an interpreter that cannot run the registry
os.execv(REAL, [REAL, "-S", *args])
"""


@pytest.fixture
def host(tmp_path):
    """The real scripts' boundary: host CLIs that only log their argv, `ss` that
    reports the gateway listening (never probe the shared 18789), a `curl` that
    fails (no model server is ever dialled), and the wrapped interpreter."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("hermes", "openclaw", "curl", "uv"):
        (bindir / name).write_text(f"#!{sys.executable}\n" + _LOGGING_CLI)
    (bindir / "ss").write_text("#!/bin/sh\necho 'LISTEN 0 4096 127.0.0.1:18789 0.0.0.0:*'\n")
    (bindir / "python").write_text(f"#!{sys.executable}\nREAL = {sys.executable!r}\n" + _PYTHON)
    for path in bindir.iterdir():
        path.chmod(0o755)
    (tmp_path / "home").mkdir()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CASCADE_", "CUDA_")) and k not in {"DISPLAY", "PYTHONPATH", "PY", *SECRETS}}
    env.update(PATH=str(bindir) + os.pathsep + os.environ["PATH"], HOME=str(tmp_path / "home"),
               PY=str(bindir / "python"), HOST_LOG=str(tmp_path / "host.jsonl"),
               B68_EXIT_CURL="7", B68_EXIT_UV="1")
    return {"env": env, "tmp": tmp_path, "log": tmp_path / "host.jsonl"}


def _run(host, script, *args, shell=None, bash="bash"):
    log = host["log"]
    log.unlink(missing_ok=True)
    result = subprocess.run([bash, str(SCRIPTS / script), *BASE_ARGS[script], *args],
                            env={**host["env"], **(shell or {})}, cwd=host["tmp"],
                            capture_output=True, text=True, timeout=120)
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, calls


def _entry(script, calls) -> list[tuple[str, str]]:
    """The (name, value) pairs the registration hands the host, in argv order."""
    if script == HERMES:
        [add] = [c for c in calls if c[:4] == ["hermes", "mcp", "add", "cascade"]]
        pairs = add[add.index("--env") + 1:add.index("--args")]
    else:
        [add] = [c for c in calls if c[:4] == ["openclaw", "mcp", "add", "cascade"]]
        pairs = [add[i + 1] for i, arg in enumerate(add) if arg == "--env"]
    return [tuple(pair.split("=", 1)) for pair in pairs]


def _registered(host, script, *args, shell=None) -> dict[str, str]:
    result, calls = _run(host, script, *args, shell=shell)
    assert result.returncode == 0, result.stdout + result.stderr
    pairs = _entry(script, calls)
    names = [name for name, _ in pairs]
    assert len(names) == len(set(names)), names  # one value per name: no host has to pick
    return dict(pairs)


def _server(script, env) -> dict:
    """What the MCP runtime selects with ONLY the registered env (a stdio host
    discards the registering shell). Hermes' entry carries its own PYTHONPATH;
    OpenClaw blocks PYTHONPATH and relies on the editable install, which
    PYTHONPATH stands in for here."""
    if script == OPENCLAW:
        env = {**env, "PYTHONPATH": str(REPO / "src")}
    code = ("import json\nfrom cascade.config import load_demo_config\n"
            "from cascade.apps.mcp_server import _readonly_lane_enabled\n"
            "c = load_demo_config(cameras=['mock'], arm='mock', llm='mock').as_dict()\n"
            "g = c['grasp']\n"
            "ports = [g['graspgenx'].get('port'), g['hug'].get('port'), c['occupancy'].get('port')]\n"
            "print(json.dumps({'executor': g.get('executor'), 'vla_port': g['vla']['port'],\n"
            "                  'readonly_lane': _readonly_lane_enabled(c), 'ports': ports}))\n")
    child = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stderr
    return json.loads(child.stdout)


def _values(names, tmp_path: Path) -> dict[str, str]:
    """A distinct value for every name, ports from this block; one free-text value
    carries every character a shell would otherwise expand or split."""
    special = {
        "CASCADE_GRASP_EXECUTOR": "vla", "CASCADE_GRASP_BACKEND": "obb", "CASCADE_OCCUPANCY": "0",
        "CASCADE_BELIEFS": "0", "CASCADE_EPISODIC": "1", "CASCADE_PROGRAMS": "1", "CASCADE_BOOTH": "1",
        "CASCADE_STREAM": "lazy", "CASCADE_VIEW": "0", "CASCADE_MJ_VIEW": "0", "CASCADE_PREWARM": "0",
        "CASCADE_OBSERVED_FINGER_GATE": "1", "CASCADE_REQUIRE_CUDA": "0", "CASCADE_DEVICE": "cpu",
        "CASCADE_KITCHEN_CAMERA_RENDERER": "isaac", "CASCADE_HIDE_TOOLS": "reset_stop",
        "CASCADE_DETECT_CLASSES": "cup,b68 bowl", "CASCADE_MEMORY_EMBEDDER": "hash",
        "CASCADE_MEMORY_EMBEDDER_MODEL": "b68 model's \"$HOME\" `id` $(id) * ~ {a,b}\nsecond line",
        "CASCADE_MCP_READONLY_LANE": "1", "CUDA_VISIBLE_DEVICES": "", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CASCADE_MICRODUCK_ENGINE": "physx", "CASCADE_MICRODUCK_DEVICE": "cuda:0",
        "CASCADE_DETECTOR_MODEL": str(tmp_path / "b68 detector.pt"),
    }
    values, port = {}, 46920
    for name in names:
        if name in special:
            values[name] = special[name]
        elif name in PORTS:
            values[name] = PORTS[name]
        elif name.endswith("_PORT"):
            values[name], port = str(port), port + 1
        elif name.endswith("_HOST"):
            values[name] = name.lower().replace("_", "-") + ".b68.invalid"
        elif name.endswith("_SHA256"):
            values[name] = hashlib.sha256(name.encode()).hexdigest()
        elif name.endswith(("_PATH", "_DIR")):
            values[name] = str(tmp_path / name.lower())
        elif name.endswith("_URL"):
            values[name] = "http://127.0.0.1:46999/" + name.lower()
        else:
            values[name] = "b68-" + name.lower()
    assert all(int(v) in BLOCK for n, v in values.items() if n.endswith("_PORT"))
    return values


def _forwarded():
    from cascade.apps.mcp_env import FORWARDED

    return FORWARDED


def test_registration_env_puts_the_registrar_values_first_and_lets_them_win():
    from cascade.apps import mcp_env

    environ = {"CASCADE_VLA_PORT": "46911", "CASCADE_DETECTOR_MODEL": "/shell", "CASCADE_ARMS": "a,b",
               "CASCADE_MCP_TOKEN": "b68-never", "CASCADE_GRASP_EXECUTOR": "vla"}
    env = mcp_env.registration_env({"CASCADE_ARM": "mock", "CASCADE_DETECTOR_MODEL": "/own"}, environ)
    assert list(env.items()) == [("CASCADE_ARM", "mock"), ("CASCADE_DETECTOR_MODEL", "/own"),
                                 ("CASCADE_GRASP_EXECUTOR", "vla"), ("CASCADE_VLA_PORT", "46911")]


def test_the_registry_cli_prints_eval_safe_words_and_refuses_a_malformed_pair(monkeypatch, capsys):
    from cascade.apps import mcp_env

    for key in list(os.environ):
        if key.startswith(("CASCADE_", "CUDA_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("CASCADE_VLA_PORT", "46911")
    assert mcp_env.main(["A=x y", "B=it's", "C="]) == 0
    out, err = capsys.readouterr()
    assert out.splitlines() == ["'A=x y'", "'B=it'\"'\"'s'", "C=", "CASCADE_VLA_PORT=46911"]
    assert "CASCADE_VLA_PORT=46911" in err and "A=" not in err
    for bad in (["NOEQUALS"], ["=value"], ["A=1", "B"]):
        assert mcp_env.main(bad) == 2
        out, err = capsys.readouterr()
        assert out == "" and "KEY=VALUE" in err


# ── premises (true on main): the scripts are live, and what stays behind stays behind ──


def test_premise_both_scripts_are_documented_entry_points():
    # Why B68 routes the scripts instead of retiring them: users are sent to them.
    readme = (REPO / "README.md").read_text()
    assert "./scripts/hermes_demo.sh" in readme and "./scripts/openclaw_demo.sh" in readme
    assert "./scripts/hermes_demo.sh" in (SCRIPTS / "install_hermes.sh").read_text()
    assert "./scripts/hermes_demo.sh" in (SCRIPTS / "setup_agents.py").read_text()
    assert "scripts/openclaw_demo.sh" in (REPO / "docs" / "OPENCLAW_2.0_INTEGRATION_BRIEF.md").read_text()
    # launch.sh registers with OpenClaw only: nothing else gives Hermes register + test + chat.
    launch = (SCRIPTS / "launch.sh").read_text()
    assert "hermes mcp" not in launch and "oc mcp set" in launch


@pytest.mark.parametrize("script", [HERMES, OPENCLAW])
def test_selectors_internals_and_secrets_are_never_registered(host, script):
    result, calls = _run(host, script, shell={**NOT_FORWARDED_SAMPLE, **SECRETS})
    assert result.returncode == 0, result.stdout + result.stderr
    env = dict(_entry(script, calls))
    assert sorted(set(env) & (set(NOT_FORWARDED_SAMPLE) | set(SECRETS))) == []
    seen = host["log"].read_text() + result.stdout + result.stderr
    assert [name for name, value in SECRETS.items() if value in seen] == []


def test_hermes_flags_still_win_over_inherited_values(host):
    env = _registered(host, HERMES, "--detector", "/b68/flag-model.pt", "--camera", "d455f", "--arm", "mock",
                      shell={"CASCADE_DETECTOR_MODEL": "/b68/shell-model.pt", "CASCADE_CAMERAS": "b68_cams",
                             "CASCADE_ARM": "b68_arm", "DISPLAY": ":7"})
    assert env["CASCADE_DETECTOR_MODEL"] == "/b68/flag-model.pt"
    assert (env["CASCADE_CAMERAS"], env["CASCADE_ARM"], env["DISPLAY"]) == ("d455f", "mock", ":7")
    assert env["PYTHONPATH"] == str(REPO / "src")


def test_openclaw_flags_and_defaults_still_win_over_inherited_values(host):
    # An empty CASCADE_DETECT_CLASSES in the shell means "use the script's
    # default vocabulary" there (${...:-...}); the forwarded empty value must
    # not undo it.
    env = _registered(host, OPENCLAW, "--cameras", "l515", "--arm", "mock",
                      shell={"CASCADE_DETECT_CLASSES": "", "CASCADE_CAMERAS": "b68_cams", "CASCADE_ARM": "b68_arm"})
    assert env["CASCADE_DETECT_CLASSES"] == "cube,banana,bottle,cup,bowl,box,plate,toy"
    assert (env["CASCADE_CAMERAS"], env["CASCADE_ARM"]) == ("l515", "mock")
    assert "PYTHONPATH" not in env  # OpenClaw blocks it ("startup safety"): never written


#: the host calls of a default run (nothing CASCADE_* / CUDA_* / DISPLAY inherited),
#: captured from main 86373d7 under bash 5 with the same doubles
GOLDEN_CALLS = {
    HERMES: [
        ["hermes", "mcp", "remove", "cascade"],
        ["hermes", "mcp", "add", "cascade", "--command", "<TMP>/bin/python", "--env", "PYTHONPATH=<REPO>/src",
         "CASCADE_CAMERAS=d455f", "CASCADE_ARM=mock", "CASCADE_DETECTOR_MODEL=<REPO>/models/yolo11n.pt",
         "DISPLAY=:1", "YOLO_OFFLINE=True", "ULTRALYTICS_OFFLINE=True", "--args", "-m", "cascade.apps.mcp_server"],
        ["hermes", "mcp", "test", "cascade"],
        ["hermes", "chat"],
    ],
    OPENCLAW: [
        ["openclaw", "mcp", "add", "cascade", "--command", "<TMP>/bin/python", "--arg", "-m", "--arg",
         "cascade.apps.mcp_server", "--cwd", "<REPO>/models", "--connect-timeout", "120",
         "--env", "CASCADE_CAMERAS=isaac,isaac_side", "--env", "CASCADE_ARM=isaac",
         "--env", "CASCADE_DETECTOR_MODEL=<REPO>/models/yoloe-11s-seg.pt",
         "--env", "CASCADE_DETECT_CLASSES=cube,banana,bottle,cup,bowl,box,plate,toy",
         "--env", "YOLO_OFFLINE=True", "--env", "ULTRALYTICS_OFFLINE=True", "--env", "DISPLAY=:1"],
        ["openclaw", "config", "set", "gateway.mode", "local"],
        ["openclaw", "gateway", "install"],
        ["openclaw", "gateway", "restart"],
        ["openclaw", "config", "validate"],
        ["openclaw", "mcp", "doctor", "cascade", "--probe"],
    ],
}


def _normalized(calls, tmp_path):
    def norm(arg):
        return arg.replace(str(REPO), "<REPO>").replace(str(tmp_path), "<TMP>")

    return [[norm(arg) for arg in call] for call in calls]


@pytest.mark.parametrize("script", [HERMES, OPENCLAW])
def test_default_registration_is_byte_identical(host, script):
    # Golden: with nothing to forward, every host call is the one main made,
    # argument for argument (including hermes_demo.sh reaching `hermes chat`).
    result, calls = _run(host, script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _normalized(calls, host["tmp"]) == GOLDEN_CALLS[script]


def test_hermes_demo_reaches_the_chat_under_the_system_bash(host):
    # macOS /bin/bash is 3.2, where "${A[@]}" of an EMPTY array under `set -u`
    # is "unbound variable": hermes_demo.sh registered, tested, then died at
    # `exec hermes chat` whenever neither --model nor --provider was given
    # (measured with a bash-3.2.57 build). /bin/bash is that shell on macOS CI.
    result, calls = _run(host, HERMES, bash="/bin/bash")
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls[-1] == ["hermes", "chat"]
    result, calls = _run(host, HERMES, "--model", "b68/model", "--provider", "b68", bash="/bin/bash")
    assert result.returncode == 0 and calls[-1] == ["hermes", "chat", "-m", "b68/model", "--provider", "b68"]


# ── behaviour (RED on main): the switches reach the entry and the server selects them ──


@pytest.mark.parametrize("script", [HERMES, OPENCLAW])
def test_the_vla_executor_and_the_readonly_lane_reach_the_server(host, script):
    env = _registered(host, script, shell={**PORTS, **SWITCHES})
    assert {name: env.get(name) for name in SWITCHES} == SWITCHES
    server = _server(script, env)
    assert server == {"executor": "vla", "vla_port": 46911, "readonly_lane": True,
                      "ports": [46901, 46904, 46902]}
    assert all(port in BLOCK for port in server["ports"])


@pytest.mark.parametrize("script", [HERMES, OPENCLAW])
def test_only_what_is_set_is_registered(host, script):
    # An unset switch is absent, never a default: the policy port set, the executor not.
    env = _registered(host, script, shell={**PORTS, "CASCADE_VLA_PORT": "46911"})
    assert env["CASCADE_VLA_PORT"] == "46911" and "CASCADE_GRASP_EXECUTOR" not in env
    assert _server(script, env)["executor"] == "analytic"


@pytest.mark.parametrize("script", [HERMES, OPENCLAW])
def test_every_registry_variable_reaches_the_entry_verbatim(host, script):
    forwarded = _forwarded()
    values = _values(forwarded, host["tmp"])
    result, calls = _run(host, script, shell=values)
    assert result.returncode == 0, result.stdout + result.stderr
    pairs = _entry(script, calls)
    env = dict(pairs)
    assert {name: env.get(name) for name in values} == values  # empty CUDA visibility included
    # the script's own values first, in its old order; the inherited ones after, in registry order
    own = [name for name in OWN[script]]
    assert [name for name, _ in pairs][:len(own)] == own
    assert [name for name, _ in pairs][len(own):] == [n for n in forwarded if n not in own]


@pytest.mark.parametrize("script", [HERMES, OPENCLAW])
def test_the_script_names_what_it_copied_from_the_shell(host, script):
    # The note must not claim a value the script replaced with its own.
    args = ["--detector", "/b68/flag-model.pt"] if script == HERMES else []
    shell = {**SWITCHES, "CASCADE_DETECTOR_MODEL": "/b68/shell-model.pt"}
    result, _ = _run(host, script, *args, shell=shell)
    assert result.returncode == 0, result.stdout + result.stderr
    [line] = [ln for ln in (result.stdout + result.stderr).splitlines() if "forwarded from this shell" in ln]
    named = {token.split("=", 1)[0] for token in line.split(": ", 1)[1].split()}
    assert named == set(SWITCHES)
    assert "CASCADE_GRASP_EXECUTOR=vla" in line


@pytest.mark.parametrize("script", [HERMES, OPENCLAW])
def test_a_registry_failure_registers_nothing(host, script):
    # Fail closed: an interpreter that cannot run the registry stops the script
    # before any host call -- hermes_demo.sh would otherwise have removed the
    # old entry already, and a fixed-env entry would drop the switches silently.
    result, calls = _run(host, script, shell={**SWITCHES, "B68_REGISTRY_FAILS": "1"})
    assert result.returncode != 0
    assert [call for call in calls if call[:2] in (["hermes", "mcp"], ["openclaw", "mcp"])] == []
    # ...stopped by set -e on the failed registry, not by bash 3.2 tripping over
    # an empty env array further down (that would be luck, and bash 5 has none)
    assert "unbound variable" not in result.stderr


@pytest.mark.parametrize("script", [HERMES, OPENCLAW])
def test_the_scripts_read_the_registry_and_keep_no_list(script):
    text = (SCRIPTS / script).read_text()
    assert '-m cascade.apps.mcp_env' in text
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert sorted(name for name in _forwarded() if name not in OWN[script] and name in code) == []


@pytest.mark.parametrize("script", [HERMES, OPENCLAW])
def test_the_scripts_stay_bash_3_2_compatible(script):
    # macOS /bin/bash 3.2: no heredoc with an odd quote count inside $( ... ),
    # no bash-4 features, and no "${A[@]}" of a possibly empty array under set -u.
    text = (SCRIPTS / script).read_text()
    assert [line for line, body in _command_substitution_heredocs(text)
            if body.count("'") % 2 or body.count('"') % 2 or body.count("(") != body.count(")")] == []
    bash4 = re.compile(r"declare -A|local -A|mapfile|readarray|\$\{\w+(,,|\^\^)\}|\|&|&>>|coproc")
    assert bash4.findall(text) == []
    assert re.findall(r'(?<!\+)"\$\{CHAT_ARGS\[@\]\}"', text) == []
