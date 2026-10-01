"""MCP registration preserves selected devices even with a stripped host env."""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys

import pytest

from conftest import REPO

sys.path.insert(0, str(REPO / "scripts"))
from setup_agents import openclaw_command, server_env  # noqa: E402


KEYS = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CASCADE_DEVICE", "CASCADE_REQUIRE_CUDA")
SELECTIONS = [
    {},
    {"CUDA_VISIBLE_DEVICES": ""},
    {"CUDA_VISIBLE_DEVICES": "1"},
    {"CUDA_VISIBLE_DEVICES": "1,0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CASCADE_DEVICE": "cuda:0"},
    {"CUDA_VISIBLE_DEVICES": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"},
    {"CUDA_VISIBLE_DEVICES": "MIG-GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/1/0"},
    {"CASCADE_DEVICE": "cpu", "CASCADE_REQUIRE_CUDA": "0"},
    {"CUDA_VISIBLE_DEVICES": "1", "CASCADE_DEVICE": "cuda:0", "CASCADE_REQUIRE_CUDA": "1"},
    {key: "" for key in KEYS},
]


@pytest.fixture
def selected_environment(monkeypatch):
    def select(values):
        for key in KEYS:
            monkeypatch.delenv(key, raising=False)
        for key, value in values.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("UNRELATED_CREDENTIAL", "must-not-forward")
    return select


def registration_env(monkeypatch, capsys):
    launch = REPO / "scripts/launch.sh"
    blocks = re.findall(r"<<'PYEOF'[^\n]*\n(.*?)\nPYEOF", launch.read_text(), re.DOTALL)
    [registration] = [block for block in blocks if '"requestTimeoutMs"' in block]
    monkeypatch.setattr(sys, "argv", ["-", sys.executable, str(REPO), "isaac", "isaac_kitchen_gpu",
                                     "/fixture/model", "", "isaac", "/fixture/state", "owner", "0", "none"])
    exec(compile(registration, str(launch), "exec"), {})
    return json.loads(capsys.readouterr().out)["env"]


@pytest.mark.parametrize("selection", SELECTIONS)
@pytest.mark.parametrize("entrypoint", ["launch", "setup_agents"])
def test_device_selection_survives_explicit_registration_and_clean_child(
    monkeypatch, capsys, selected_environment, selection, entrypoint,
):
    selected_environment(selection)
    env = (registration_env(monkeypatch, capsys) if entrypoint == "launch"
           else server_env("isaac", "isaac_kitchen_gpu", ":1"))
    assert {key: env[key] for key in KEYS if key in env} == selection
    assert "UNRELATED_CREDENTIAL" not in env
    # Model the stdio host discarding inherited variables: only explicit env
    # reaches this stdlib child. No CUDA import, device context or robot RPC.
    child = subprocess.run(
        [sys.executable, "-c", "import json,os,sys; print(json.dumps({k:os.environ[k] for k in sys.argv[1:] if k in os.environ}))", *KEYS],
        env=env, text=True, capture_output=True, timeout=5, check=True,
    )
    assert json.loads(child.stdout) == selection


@pytest.mark.parametrize("selection", SELECTIONS)
def test_openclaw_command_keeps_explicit_devices_and_omits_pythonpath(selected_environment, selection):
    selected_environment(selection)
    tokens = shlex.split(openclaw_command(sys.executable, server_env("mock", "mock", ":1")))
    env = dict(tokens[i + 1].split("=", 1) for i, token in enumerate(tokens) if token == "--env")
    assert {key: env[key] for key in KEYS if key in env} == selection
    assert "PYTHONPATH" not in env


@pytest.mark.parametrize("key", KEYS)
def test_explicit_setup_override_wins_including_empty(selected_environment, key):
    selected_environment({key: "inherited"})
    assert server_env("mock", "mock", ":1", extra=[f"{key}=explicit"])[key] == "explicit"
    assert server_env("mock", "mock", ":1", extra=[f"{key}="])[key] == ""
