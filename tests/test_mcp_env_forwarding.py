"""B63: every runtime switch reaches the MCP server a registration writes.

A stdio MCP host (OpenClaw, Hermes, Codex, Claude) starts
`python -m cascade.apps.mcp_server` with the `env` its registration names, not
with the shell that registered it. Before B63 `scripts/launch.sh` copied a
FIXED list of 18 names into the `mcp set` JSON -- without
`CASCADE_GRASP_EXECUTOR` / `CASCADE_VLA_PORT` (B49) -- and
`scripts/setup_agents.py` used its own rule (four device names + every
`CASCADE_*_PORT` / `_HOST`). So docs/VLA_EXECUTOR.md's
`CASCADE_GRASP_EXECUTOR=vla CASCADE_VLA_PORT=<p> ./run.sh <profile>` registered
a server that ran the analytic executor, while the launcher's own runtime
check, which inherits the whole shell, saw `vla`.

`cascade.apps.mcp_env` is now the one registry both registrations read: every
`CASCADE_*` variable the runtime (`src/cascade`) reads is FORWARDED (copied
verbatim when present, never invented) or NOT_FORWARDED with a category and a
reason. The guard below fails when the runtime reads a new one that is in
neither. Behaviour is measured through the REAL launch.sh (the
`launcher_boundary` harness: `mcp set` JSON written to MCP_CONFIG) and the
real setup_agents.py. Ports are from this item's block (46500-46599); nothing
here dials them.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO
from test_launch_delivery import launcher_boundary as _launcher_boundary
from test_launch_delivery import model_http_boundary as _model_http_boundary

sys.path.insert(0, str(REPO / "scripts"))
import setup_agents  # noqa: E402

launcher_boundary = _launcher_boundary  # the real launch.sh; only host CLI/model HTTP/deps doubled
model_http_boundary = _model_http_boundary

SRC = REPO / "src" / "cascade"
LAUNCH = REPO / "scripts" / "launch.sh"
NAME = re.compile(r"CASCADE_[A-Z0-9_]+")
#: names that would be a credential; none may ever be forwarded
SECRET = re.compile(r"TOKEN|SECRET|PASSW|CREDENTIAL|API_KEY|(?:^|_)KEY$")
#: this item's port block; the launcher's sidecar ports are always set inside it
PORTS = {"CASCADE_GRASPGENX_PORT": "46501", "CASCADE_OCCUPANCY_PORT": "46502",
         "CASCADE_BRIDGE_PORT": "46503", "CASCADE_HUG_PORT": "46504"}
#: what launcher_boundary itself sets; everything else CASCADE_* is scrubbed
HARNESS = {"CASCADE_OPENCLAW_PROFILE", "CASCADE_LAUNCH_STATE", "CASCADE_COSMOS_BASE_URL",
           "CASCADE_QWEN_BASE_URL", "CASCADE_MCP_NAME"}
#: read by the runtime, must never be copied from the registering shell (one per category)
NOT_FORWARDED_SAMPLE = {
    "CASCADE_ARMS": "so101_left,so101_right",   # would replace the arm the registration names
    "CASCADE_CAMERA": "mock_rgb",
    "CASCADE_ROBOT": "b63_robot",               # a composed robot instead of the named rig
    "CASCADE_BASE": "microduck_isaac",
    "CASCADE_RUN_DIR": "/b63/one-shared-run-dir",  # per-process, derived by the server
    "CASCADE_LLM": "openai",                    # demo CLI brain; the MCP host is the brain
    "CASCADE_CODEX_BIN": "/b63/codex",
    "CASCADE_PREMOTION_CHECK": "0",
    "CASCADE_JUDGE": "fake",                    # launcher-side judge pass (--no-judge wins)
    "CASCADE_JUDGE_TIMEOUT_S": "7",
    "CASCADE_MCP_SSE_KEEPALIVE_S": "3",         # HTTP transport only
}
#: credentials a launcher shell may carry; their VALUES must appear nowhere
SECRETS = {"CASCADE_MCP_TOKEN": "b63-mcp-bearer-0b9e1c", "CASCADE_MODEL_API_KEY": "b63-model-key-77d1",
           "HF_TOKEN": "hf_b63_secret_4a2f", "GITHUB_TOKEN": "ghp_b63_secret_91c3"}


def _registry():
    from cascade.apps import mcp_env

    return mcp_env


# ── what the runtime reads (the guard's input) ────────────────────────────


def _runtime_literals() -> dict[str, set[str]]:
    """Every string constant in src/cascade that is a whole CASCADE_* name -> files.
    The registry itself is not a reader (its keys would make every entry look read)."""
    found: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        if path == SRC / "apps" / "mcp_env.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and NAME.fullmatch(node.value):
                found.setdefault(node.value, set()).add(str(path.relative_to(SRC)))
    return found


def _families() -> dict[str, list[str]]:
    """Names composed at run time: a prefix literal -> the names its reader builds."""
    from cascade import config

    return {config.MICRODUCK_ENV_PREFIX: [config.MICRODUCK_ENV_PREFIX + key.upper()
                                          for key in config.MICRODUCK_ENV_KEYS]}


def _runtime_reads() -> set[str]:
    literals, families, names = _runtime_literals(), _families(), set()
    for name, files in literals.items():
        if name.endswith("_"):
            assert name in families, (
                f"{name}* is composed in {sorted(files)}: list its members in _families() "
                "and classify each in cascade.apps.mcp_env")
            names.update(families[name])
        else:
            names.add(name)
    return names


def test_premise_the_scan_sees_the_reads_this_item_is_about():
    # Premise (true on main): the AST scan finds the B49 switches, the composed
    # MicroDuck family and the one secret, so the guard below has teeth.
    literals = _runtime_literals()
    assert {"CASCADE_GRASP_EXECUTOR", "CASCADE_VLA_PORT", "CASCADE_MCP_TOKEN", "CASCADE_ARMS"} <= set(literals)
    assert "config.py" in literals["CASCADE_GRASP_EXECUTOR"]
    assert "CASCADE_MICRODUCK_" in literals
    assert len(literals) >= 50


# ── the registry (one source of truth) ────────────────────────────────────


def test_every_cascade_variable_the_runtime_reads_is_classified():
    reg = _registry()
    unclassified = sorted(_runtime_reads() - set(reg.FORWARDED) - set(reg.NOT_FORWARDED))
    assert not unclassified, (
        "the runtime reads these but cascade.apps.mcp_env neither forwards them to the MCP "
        f"server nor lists them as not forwarded with a reason: {unclassified}")
    assert not set(reg.FORWARDED) & set(reg.NOT_FORWARDED)


def test_no_registry_entry_is_stale():
    reg = _registry()
    reads = _runtime_reads()
    listed = [*reg.FORWARDED, *reg.NOT_FORWARDED]
    assert sorted(n for n in listed if n.startswith("CASCADE_") and n not in reads) == []
    # The only non-CASCADE names: the CUDA device selection, kept verbatim.
    assert [n for n in listed if not n.startswith("CASCADE_")] == ["CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"]


def test_every_entry_says_what_it_is():
    reg = _registry()
    assert set(reg.CATEGORIES) == {"registration", "selection", "internal", "cli", "launcher", "transport",
                                   "secret"}
    for name, (category, reason) in reg.NOT_FORWARDED.items():
        assert category in reg.CATEGORIES, name
        assert isinstance(reason, str) and len(reason) >= 30, name
    for name, what in reg.FORWARDED.items():
        assert isinstance(what, str) and len(what) >= 10, name


def test_nothing_secret_is_ever_forwarded():
    reg = _registry()
    assert [n for n in reg.FORWARDED if SECRET.search(n)] == []
    secret_reads = sorted(n for n in _runtime_reads() if SECRET.search(n))
    assert "CASCADE_MCP_TOKEN" in secret_reads
    assert {n: reg.NOT_FORWARDED[n][0] for n in secret_reads} == {n: "secret" for n in secret_reads}
    # forwarded_env copies registry names only: a secret in the environment stays behind.
    assert reg.forwarded_env({**SECRETS, "OPENAI_API_KEY": "sk-b63"}) == {}


def test_the_not_forwarded_sample_is_classified_by_category():
    reg = _registry()
    categories = {name: reg.NOT_FORWARDED[name][0] for name in [*NOT_FORWARDED_SAMPLE, "CASCADE_MCP_TOKEN"]}
    assert categories == {
        "CASCADE_ARMS": "selection", "CASCADE_CAMERA": "selection", "CASCADE_ROBOT": "selection",
        "CASCADE_BASE": "selection", "CASCADE_RUN_DIR": "internal", "CASCADE_LLM": "cli",
        "CASCADE_CODEX_BIN": "cli", "CASCADE_PREMOTION_CHECK": "cli", "CASCADE_JUDGE": "launcher",
        "CASCADE_JUDGE_TIMEOUT_S": "launcher", "CASCADE_MCP_SSE_KEEPALIVE_S": "transport",
        "CASCADE_MCP_TOKEN": "secret"}
    assert {reg.NOT_FORWARDED[n][0] for n in ("CASCADE_CAMERAS", "CASCADE_ARM", "CASCADE_OPENCLAW_PROFILE")} == {
        "registration"}


def test_the_b41_endpoints_and_the_b49_switches_are_forwarded():
    from cascade.config import ENDPOINT_ENV_VARS, VLA_PORT_ENV

    forwarded = set(_registry().FORWARDED)
    assert set(ENDPOINT_ENV_VARS) <= forwarded
    assert {VLA_PORT_ENV, "CASCADE_GRASP_EXECUTOR", "CASCADE_GRASP_BACKEND"} <= forwarded


def test_forwarded_env_copies_present_names_verbatim_in_registry_order():
    reg = _registry()
    names = list(reg.FORWARDED)
    environ = {names[5]: "", names[2]: " kept as given ", "UNRELATED": "x", names[0]: "a"}
    assert list(reg.forwarded_env(environ).items()) == [(names[0], "a"), (names[2], " kept as given "),
                                                         (names[5], "")]
    assert reg.forwarded_env(environ, skip={names[2]}) == {names[0]: "a", names[5]: ""}
    assert reg.forwarded_env({}) == {}


def test_forwarded_env_reads_the_process_environment_by_default(monkeypatch):
    monkeypatch.setenv("CASCADE_GRASP_EXECUTOR", "vla")
    monkeypatch.setenv("CASCADE_VLA_PORT", "46511")
    got = _registry().forwarded_env()
    assert (got["CASCADE_GRASP_EXECUTOR"], got["CASCADE_VLA_PORT"]) == ("vla", "46511")


def test_the_forwarded_set_is_pinned():
    # A reclassification (a switch silently no longer forwarded, or a selector
    # suddenly forwarded) must show up here, in review, not in a live run.
    assert set(_registry().FORWARDED) == {
        "CASCADE_GRASP_MEMORY_PATH", "CASCADE_ENVELOPE_PATH", "CASCADE_BELIEFS_PATH", "CASCADE_BELIEFS",
        "CASCADE_GRASP_BACKEND", "CASCADE_GRASPGENX_PORT", "CASCADE_GRASPGENX_HOST", "CASCADE_BRIDGE_PORT",
        "CASCADE_OCCUPANCY_PORT", "CASCADE_HUG_PORT", "CASCADE_HUG_HOST", "CASCADE_GRASP_EVIDENCE_DIR",
        "CASCADE_OBSERVED_FINGER_GATE", "CASCADE_KITCHEN_CAMERA_RENDERER", "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER", "CASCADE_DEVICE", "CASCADE_REQUIRE_CUDA", "CASCADE_OCCUPANCY",
        "CASCADE_GRASP_EXECUTOR", "CASCADE_VLA_PORT", "CASCADE_BOOTH", "CASCADE_DETECTOR_MODEL",
        "CASCADE_DETECT_CLASSES", "CASCADE_HIDE_TOOLS", "CASCADE_STREAM", "CASCADE_STREAM_PORT", "CASCADE_VIEW",
        "CASCADE_MJ_VIEW", "CASCADE_PREWARM", "CASCADE_EXTERNAL_VIEW_URL", "CASCADE_EPISODIC",
        "CASCADE_EPISODIC_PATH", "CASCADE_PROGRAMS", "CASCADE_PROGRAMS_PATH", "CASCADE_MEMORY_EMBEDDER",
        "CASCADE_MEMORY_EMBEDDER_MODEL", "CASCADE_MOTION_EVIDENCE_DIR", "CASCADE_GPU_PERCEPTION_EVIDENCE_DIR",
        "CASCADE_MICRODUCK_ASSET_SHA256", "CASCADE_MICRODUCK_POLICY_SHA256",
        "CASCADE_MICRODUCK_MODEL_IDENTITY_SHA256", "CASCADE_MICRODUCK_BRIDGE_PORT", "CASCADE_MICRODUCK_ENGINE",
        "CASCADE_MICRODUCK_DEVICE"}


def test_the_old_launch_order_comes_first():
    # Byte-identical registrations need the 18 names launch.sh forwarded before
    # B63 first, in their old order (new names are appended after them).
    assert list(_registry().FORWARDED)[:18] == [
        "CASCADE_GRASP_MEMORY_PATH", "CASCADE_ENVELOPE_PATH", "CASCADE_BELIEFS_PATH", "CASCADE_BELIEFS",
        "CASCADE_GRASP_BACKEND", "CASCADE_GRASPGENX_PORT", "CASCADE_GRASPGENX_HOST",
        "CASCADE_BRIDGE_PORT", "CASCADE_OCCUPANCY_PORT", "CASCADE_HUG_PORT", "CASCADE_HUG_HOST",
        "CASCADE_GRASP_EVIDENCE_DIR", "CASCADE_OBSERVED_FINGER_GATE", "CASCADE_KITCHEN_CAMERA_RENDERER",
        "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CASCADE_DEVICE", "CASCADE_REQUIRE_CUDA"]


# ── helpers for the real registrations ────────────────────────────────────


def _values(names, tmp_path: Path) -> dict[str, str]:
    """A distinct value each runtime would accept for every name (ports from this block)."""
    special = {
        "CASCADE_GRASP_EXECUTOR": "vla", "CASCADE_GRASP_BACKEND": "obb", "CASCADE_OCCUPANCY": "0",
        "CASCADE_BELIEFS": "0", "CASCADE_EPISODIC": "1", "CASCADE_PROGRAMS": "1", "CASCADE_BOOTH": "1",
        "CASCADE_STREAM": "lazy", "CASCADE_VIEW": "0", "CASCADE_MJ_VIEW": "0", "CASCADE_PREWARM": "0",
        "CASCADE_OBSERVED_FINGER_GATE": "1", "CASCADE_REQUIRE_CUDA": "0", "CASCADE_DEVICE": "cpu",
        "CASCADE_KITCHEN_CAMERA_RENDERER": "isaac", "CASCADE_HIDE_TOOLS": "reset_stop",
        "CASCADE_DETECT_CLASSES": "cup,b63 bowl", "CASCADE_MEMORY_EMBEDDER": "hash",
        "CUDA_VISIBLE_DEVICES": "", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CASCADE_MICRODUCK_ENGINE": "physx", "CASCADE_MICRODUCK_DEVICE": "cuda:0",
    }
    detector = tmp_path / "b63-detector.pt"
    detector.write_text("b63 placeholder; never loaded")  # launch.sh refuses a missing weights file
    special["CASCADE_DETECTOR_MODEL"] = str(detector)
    values, port = {}, 46520
    for name in names:
        if name in special:
            values[name] = special[name]
        elif name in PORTS:
            values[name] = PORTS[name]
        elif name.endswith("_PORT"):
            values[name], port = str(port), port + 1
        elif name.endswith("_HOST"):
            values[name] = name.lower().replace("_", "-") + ".b63.invalid"
        elif name.endswith("_SHA256"):
            values[name] = hashlib.sha256(name.encode()).hexdigest()
        elif name.endswith(("_PATH", "_DIR")):
            values[name] = str(tmp_path / name.lower())
        elif name.endswith("_URL"):
            values[name] = "http://127.0.0.1:46599/" + name.lower()
        else:
            values[name] = "b63-" + name.lower()
    return values


def _scrubbed(h: dict) -> dict:
    """launcher_boundary with no inherited CASCADE_* / CUDA_* / DISPLAY, ports in this block."""
    env = h["env"]
    for key in list(env):
        if (key.startswith("CASCADE_") and key not in HARNESS) or key.startswith("CUDA_") or key == "DISPLAY":
            del env[key]
    env.update(PORTS)
    return h


def _launch(h: dict) -> tuple[str, dict]:
    result = subprocess.run(h["command"], env=h["env"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    text = Path(h["env"]["MCP_CONFIG"]).read_text()
    return text, json.loads(text)


def _normalized(text: str, tmp_path: Path) -> str:
    text = text.replace(str(tmp_path), "<TMP>")
    return re.sub(r'"--launch-owner", "[0-9a-f]{32}"', '"--launch-owner", "<OWNER>"', text)


def _server_grasp(env: dict) -> dict:
    """What the MCP runtime's config loader selects with ONLY this env (the stdio host
    discards the registering shell; PYTHONPATH stands in for the editable install)."""
    code = ("import json\nfrom cascade.config import load_demo_config\n"
            "g = load_demo_config(cameras=['mock'], arm='mock', llm='mock').as_dict()['grasp']\n"
            "print(json.dumps({'executor': g.get('executor'), 'vla_port': g['vla']['port']}))\n")
    child = subprocess.run([sys.executable, "-c", code], env={**env, "PYTHONPATH": str(REPO / "src")},
                           capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stderr
    return json.loads(child.stdout)


def _in_process_registration(monkeypatch, capsys, *, sim="none", detector="/fixture/model", classes="",
                             gui="0", occupancy="auto") -> dict:
    """launch.sh's registration heredoc, executed verbatim in this process."""
    blocks = re.findall(r"<<'PYEOF'[^\n]*\n(.*?)\nPYEOF", LAUNCH.read_text(), re.DOTALL)
    [registration] = [block for block in blocks if '"requestTimeoutMs"' in block]
    monkeypatch.setattr(sys, "argv", ["-", "/fixture/python", "/fixture/repo", "mock", "mock", detector, classes,
                                      sim, "/fixture/state", "f" * 32, gui, occupancy])
    exec(compile(registration, str(LAUNCH), "exec"), {})
    return json.loads(capsys.readouterr().out)


def _scrub_process(monkeypatch):
    for key in list(os.environ):
        if key.startswith(("CASCADE_", "CUDA_")) or key == "DISPLAY":
            monkeypatch.delenv(key, raising=False)
    for key, value in PORTS.items():
        monkeypatch.setenv(key, value)


# ── the premise: the stdio host starts the server with the registered env only ──


def test_premise_the_runtime_selects_the_executor_from_the_server_environment(monkeypatch):
    # True on main (B49): with the variables in the SERVER's environment the
    # runtime selects the VLA executor on the policy port; without, the analytic
    # one on the configured port. So the defect is purely what the
    # registration copies, not the runtime.
    _scrub_process(monkeypatch)
    base = {k: v for k, v in os.environ.items() if not k.startswith(("CASCADE_", "CUDA_"))}
    assert _server_grasp({**base, **PORTS}) == {"executor": "analytic", "vla_port": 8000}
    assert _server_grasp({**base, **PORTS, "CASCADE_GRASP_EXECUTOR": "vla", "CASCADE_VLA_PORT": "46511"}) == {
        "executor": "vla", "vla_port": 46511}


# ── launch.sh (real script) ───────────────────────────────────────────────


def test_launch_registers_the_vla_executor_and_its_server_selects_it(launcher_boundary):
    h = _scrubbed(launcher_boundary)
    h["env"].update(CASCADE_GRASP_EXECUTOR="vla", CASCADE_VLA_PORT="46511")
    env = _launch(h)[1]["env"]
    assert (env.get("CASCADE_GRASP_EXECUTOR"), env.get("CASCADE_VLA_PORT")) == ("vla", "46511")
    assert _server_grasp(env) == {"executor": "vla", "vla_port": 46511}


def test_launch_forwards_every_registry_variable_verbatim(launcher_boundary, tmp_path):
    reg = _registry()
    h = _scrubbed(launcher_boundary)
    values = _values(reg.FORWARDED, tmp_path)
    h["env"].update(values)
    env = _launch(h)[1]["env"]
    assert {name: env.get(name) for name in values} == values


def test_launch_registers_only_what_is_set(launcher_boundary):
    # An unset switch is absent, never a default: one of the pair set, the other not.
    h = _scrubbed(launcher_boundary)
    h["env"]["CASCADE_VLA_PORT"] = "46511"
    env = _launch(h)[1]["env"]
    assert env.get("CASCADE_VLA_PORT") == "46511" and "CASCADE_GRASP_EXECUTOR" not in env
    assert _server_grasp(env) == {"executor": "analytic", "vla_port": 46511}


def test_launch_never_registers_selectors_internals_or_secrets(launcher_boundary):
    # Premise-shaped (also true on main): what must stay behind stays behind.
    h = _scrubbed(launcher_boundary)
    h["env"].update(NOT_FORWARDED_SAMPLE)
    h["env"].update(SECRETS)
    text, config = _launch(h)
    env = config["env"]
    assert sorted(set(NOT_FORWARDED_SAMPLE) & set(env)) == []
    assert sorted(set(SECRETS) & set(env)) == []
    log = Path(h["env"]["HOST_LOG"]).read_text()
    for secret in SECRETS.values():
        assert secret not in text and secret not in log
    assert (env["CASCADE_CAMERAS"], env["CASCADE_ARM"]) == ("mock", "mock")  # the flags, not CASCADE_ARMS


def test_launch_flags_still_win_over_inherited_values(launcher_boundary):
    # True on main: --occupancy none / --graspgenx none and the launcher's own
    # detector resolution beat an inherited value of the same variable.
    h = _scrubbed(launcher_boundary)
    h["env"].update(CASCADE_OCCUPANCY="1", CASCADE_GRASP_BACKEND="graspgenx", CASCADE_DETECTOR_MODEL="",
                    CASCADE_CAMERAS="b63_cameras", CASCADE_ARM="b63_arm")
    env = _launch(h)[1]["env"]
    assert env["CASCADE_OCCUPANCY"] == "0"
    assert env["CASCADE_GRASP_BACKEND"] == "obb"
    assert env["CASCADE_DETECTOR_MODEL"] == str(h["repo"] / "models" / "yoloe-11s-seg.pt")
    assert (env["CASCADE_CAMERAS"], env["CASCADE_ARM"]) == ("mock", "mock")


#: launch.sh --sim none --arm mock --cameras mock --occupancy none --graspgenx none,
#: profile isolated-test, nothing else inherited (captured from main be57535)
GOLDEN_LAUNCH = (
    '{"command": "<TMP>/bin/python", "args": ["-m", "cascade.apps.mcp_server", "--launch-owner", '
    '"<OWNER>", "--launch-state-dir", "<TMP>/state/profile-isolated-test"], '
    '"cwd": "<TMP>/repo/models", "env": {"CASCADE_CAMERAS": "mock", "CASCADE_ARM": "mock", '
    '"CASCADE_DETECTOR_MODEL": "<TMP>/repo/models/yoloe-11s-seg.pt", "YOLO_OFFLINE": "True", '
    '"ULTRALYTICS_OFFLINE": "True", "CASCADE_OPENCLAW_PROFILE": "isolated-test", '
    '"CASCADE_GRASP_MEMORY_PATH": "<TMP>/state/profile-isolated-test/memory/grasp_memory.json", '
    '"CASCADE_ENVELOPE_PATH": "<TMP>/state/profile-isolated-test/memory/envelope.json", '
    '"CASCADE_BELIEFS_PATH": "<TMP>/state/profile-isolated-test/memory/beliefs.json", '
    '"CASCADE_GRASP_BACKEND": "obb", "CASCADE_GRASPGENX_PORT": "46501", '
    '"CASCADE_BRIDGE_PORT": "46503", "CASCADE_OCCUPANCY_PORT": "46502", '
    '"CASCADE_HUG_PORT": "46504", "CASCADE_OCCUPANCY": "0"}, "connectionTimeoutMs": 120000, '
    '"requestTimeoutMs": 300000}')


def test_launch_default_registration_is_byte_identical(launcher_boundary, tmp_path):
    h = _scrubbed(launcher_boundary)
    text, _ = _launch(h)
    assert _normalized(text, tmp_path) == GOLDEN_LAUNCH


#: the registration heredoc for the Isaac/Spark shape (sim isaac, no GUI,
#: occupancy none) with the Spark's CUDA requirement, captured from main be57535
GOLDEN_ISAAC = (
    '{"command": "/fixture/python", "args": ["-m", "cascade.apps.mcp_server", "--launch-owner", '
    '"ffffffffffffffffffffffffffffffff", "--launch-state-dir", "/fixture/state"], '
    '"cwd": "/fixture/repo/models", "env": {"CASCADE_CAMERAS": "mock", "CASCADE_ARM": "mock", '
    '"CASCADE_DETECTOR_MODEL": "/fixture/model", "YOLO_OFFLINE": "True", '
    '"ULTRALYTICS_OFFLINE": "True", "CASCADE_OPENCLAW_PROFILE": "cascade-demo", '
    '"CASCADE_GRASPGENX_PORT": "46501", "CASCADE_BRIDGE_PORT": "46503", '
    '"CASCADE_OCCUPANCY_PORT": "46502", "CASCADE_HUG_PORT": "46504", '
    '"CASCADE_REQUIRE_CUDA": "1", "CASCADE_OCCUPANCY": "0", "CASCADE_VIEW": "0"}, '
    '"connectionTimeoutMs": 120000, "requestTimeoutMs": 300000}')


def test_isaac_registration_is_byte_identical(monkeypatch, capsys):
    _scrub_process(monkeypatch)
    monkeypatch.setenv("CASCADE_REQUIRE_CUDA", "1")
    monkeypatch.setenv("CASCADE_OPENCLAW_PROFILE", "cascade-demo")
    # The Spark is Linux: on macOS the heredoc adds DISPLAY=":0" for the viewer
    # (an unrelated, older branch), so pin the platform the golden describes.
    monkeypatch.setattr(sys, "platform", "linux")
    config = _in_process_registration(monkeypatch, capsys, sim="isaac", occupancy="none")
    assert json.dumps(config) == GOLDEN_ISAAC


def _command_substitution_heredocs(text: str) -> list[tuple[int, str]]:
    """(line, body) of every heredoc opened on a line that starts a `$(`."""
    lines, found, i = text.split("\n"), [], 0
    while i < len(lines):
        opened = re.search(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", lines[i])
        if opened and "$(" in lines[i]:
            end = lines.index(opened.group(1), i + 1)
            found.append((i + 1, "\n".join(lines[i + 1:end])))
            i = end
        i += 1
    return found


def test_launch_heredocs_in_command_substitutions_parse_under_bash_3_2():
    # macOS /bin/bash is 3.2, which scans a heredoc inside `$( ... )` as shell
    # text: one apostrophe in the registration heredoc's comment (B63's first
    # push) made bash 3.2 reject the whole launch.sh ("line 1131: syntax error
    # near unexpected token `('"), measured with a bash-3.2.57 build and on
    # macos-latest CI. Keep quotes and parentheses balanced in every such body.
    blocks = _command_substitution_heredocs(LAUNCH.read_text())
    assert any('"requestTimeoutMs"' in body for _, body in blocks)  # the registration is one of them
    unbalanced = [line for line, body in blocks
                  if body.count("'") % 2 or body.count('"') % 2 or body.count("(") != body.count(")")]
    assert unbalanced == []


# ── setup_agents.py (real registrar) ──────────────────────────────────────


def test_setup_agents_forwards_the_same_registry(monkeypatch, tmp_path, capsys):
    reg = _registry()
    _scrub_process(monkeypatch)
    values = _values(reg.FORWARDED, tmp_path)
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    env = setup_agents.server_env("isaac", "isaac_kitchen_gpu", ":1")
    assert {name: env.get(name) for name in values} == values
    # ...through main() into the files it writes, and named on stdout
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setattr(sys, "argv", ["setup_agents.py", "--host", "codex", "--python", "/usr/bin/python3",
                                      "--camera", "isaac", "--arm", "isaac_kitchen_gpu", "--write"])
    assert setup_agents.main() == 0
    import tomllib

    written = tomllib.loads((tmp_path / "codex" / "config.toml").read_text())["mcp_servers"]["cascade"]["env"]
    assert (written["CASCADE_GRASP_EXECUTOR"], written["CASCADE_VLA_PORT"]) == ("vla", values["CASCADE_VLA_PORT"])
    out = capsys.readouterr().out
    assert "CASCADE_GRASP_EXECUTOR=vla" in out and "forward" in out


def test_setup_agents_explicit_flags_still_win(monkeypatch):
    _scrub_process(monkeypatch)
    monkeypatch.setenv("CASCADE_GRASP_EXECUTOR", "vla")
    monkeypatch.setenv("CASCADE_HIDE_TOOLS", "camera_snapshot")
    monkeypatch.setenv("CASCADE_DETECT_CLASSES", "shell,words")
    monkeypatch.setenv("CASCADE_VLA_PORT", "not-a-port")  # given with --env: never read, never checked
    env = setup_agents.server_env("mock", "mock", ":1", detect_classes="cup", hide_tools="reset_stop",
                                  extra=["CASCADE_GRASP_EXECUTOR=analytic", "CASCADE_VLA_PORT=46512"])
    assert env["CASCADE_GRASP_EXECUTOR"] == "analytic" and env["CASCADE_VLA_PORT"] == "46512"
    assert (env["CASCADE_HIDE_TOOLS"], env["CASCADE_DETECT_CLASSES"]) == ("reset_stop", "cup")


def test_setup_agents_reports_only_what_it_copied_from_the_shell(monkeypatch, tmp_path, capsys):
    # The printed "forwarded from this shell" line must not claim a value that
    # an explicit flag replaced.
    _scrub_process(monkeypatch)
    monkeypatch.setenv("CASCADE_GRASP_EXECUTOR", "vla")
    monkeypatch.setenv("CASCADE_DETECT_CLASSES", "shell,words")
    monkeypatch.setenv("CASCADE_HIDE_TOOLS", "camera_snapshot")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setattr(sys, "argv", ["setup_agents.py", "--host", "codex", "--python", "/usr/bin/python3",
                                      "--detect-classes", "cup", "--hide-tools", "reset_stop"])
    assert setup_agents.main() == 0
    [line] = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("# forwarded from this shell")]
    named = {token.split("=", 1)[0] for token in line.split(": ", 1)[1].split()}
    assert named == {"CASCADE_GRASP_EXECUTOR", *PORTS}
    assert "CASCADE_GRASP_EXECUTOR=vla" in line


@pytest.mark.parametrize("inherited", [{}, {"CASCADE_VLA_PORT": "46511"}], ids=["nothing", "a-port"])
def test_setup_agents_runs_under_a_bare_python(tmp_path, inherited):
    # `python -S`: no site-packages, so no installed cascade and no PyYAML. The
    # registrar finds this checkout's registry itself; a port it cannot check
    # is refused, never written unchecked.
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path / "home"),
           "CODEX_HOME": str(tmp_path / "codex"), **inherited}
    result = subprocess.run([sys.executable, "-S", str(REPO / "scripts" / "setup_agents.py"), "--host", "codex",
                             "--camera", "mock", "--python", "/usr/bin/python3"],
                            env=env, capture_output=True, text=True, timeout=60)
    if inherited:
        assert result.returncode == 2 and "cannot check CASCADE_VLA_PORT" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert 'CASCADE_CAMERAS = "mock"' in result.stdout


def test_setup_agents_refuses_a_policy_port_the_runtime_would_refuse(monkeypatch):
    _scrub_process(monkeypatch)
    monkeypatch.setenv("CASCADE_VLA_PORT", "+8000")
    with pytest.raises(ValueError, match="CASCADE_VLA_PORT"):
        setup_agents.server_env("mock", "mock", ":1")


def test_setup_agents_never_writes_selectors_internals_or_secrets(monkeypatch, tmp_path):
    _scrub_process(monkeypatch)
    for name, value in {**NOT_FORWARDED_SAMPLE, **SECRETS, "OPENAI_API_KEY": "sk-b63-secret",
                        "ANTHROPIC_API_KEY": "sk-ant-b63-secret"}.items():
        monkeypatch.setenv(name, value)
    env = setup_agents.server_env("isaac", "isaac_kitchen_gpu", ":1")
    assert sorted(set(env) & (set(NOT_FORWARDED_SAMPLE) | set(SECRETS) | {"OPENAI_API_KEY"})) == []
    blocks = [setup_agents.codex_toml_block("/usr/bin/python3", env),
              setup_agents.claude_mcp_json(None, "/usr/bin/python3", env),
              setup_agents.claude_add_command("/usr/bin/python3", env),
              setup_agents.openclaw_command("/usr/bin/python3", env)]
    for secret in [*SECRETS.values(), "sk-b63-secret", "sk-ant-b63-secret"]:
        assert not [b for b in blocks if secret in b]


def test_setup_agents_default_entry_is_unchanged(monkeypatch):
    # Golden (true on main): nothing inherited -> exactly the six base entries, in order.
    _scrub_process(monkeypatch)
    for key in PORTS:
        monkeypatch.delenv(key)
    env = setup_agents.server_env("isaac", "isaac_kitchen_gpu", ":1")
    assert list(env.items()) == [("PYTHONPATH", str(REPO / "src")), ("CASCADE_CAMERAS", "isaac"),
                                 ("CASCADE_ARM", "isaac_kitchen_gpu"), ("DISPLAY", ":1"),
                                 ("YOLO_OFFLINE", "True"), ("ULTRALYTICS_OFFLINE", "True")]


# ── one source: both registrations forward the same names and values ──────


def test_launch_and_setup_agents_forward_identically(monkeypatch, capsys, tmp_path):
    reg = _registry()
    _scrub_process(monkeypatch)
    values = _values(reg.FORWARDED, tmp_path)
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    launched = _in_process_registration(monkeypatch, capsys, detector=values["CASCADE_DETECTOR_MODEL"],
                                        classes=values["CASCADE_DETECT_CLASSES"])["env"]
    registered = setup_agents.server_env("mock", "mock", ":1")
    assert {n: launched.get(n) for n in reg.FORWARDED} == {n: registered.get(n) for n in reg.FORWARDED} == values


def test_both_registrations_read_the_registry():
    # Static half of "one source": neither keeps a list of its own.
    blocks = re.findall(r"<<'PYEOF'[^\n]*\n(.*?)\nPYEOF", LAUNCH.read_text(), re.DOTALL)
    [registration] = [block for block in blocks if '"requestTimeoutMs"' in block]
    assert "from cascade.apps.mcp_env import forwarded_env" in registration
    assert "CASCADE_GRASP_MEMORY_PATH" not in registration and "CUDA_DEVICE_ORDER" not in registration
    source = (REPO / "scripts" / "setup_agents.py").read_text()
    assert "from cascade.apps.mcp_env import" in source
    assert '"CUDA_DEVICE_ORDER"' not in source and "ENDPOINT_VAR = " not in source
