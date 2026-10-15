"""scripts/booth_up.sh: the booth pre-flight's perception checks (Ruling 26).

Two checks in the script's ok/warn style: the SAM weights the robot server
loads (configs/demo.yaml `segmenter.model` = ${repo}/models/sam2.1_l.pt) and
the perception brain (`${CASCADE_PERCEPTION_URL:-http://127.0.0.1:8080/v1}/models`
answering within 2 s). Like every other check they only print and mark the
run red; the script still changes nothing.

The script runs from a copy inside a scratch repo (it finds its repo from its
own path), with a double for every outside probe: a curl that records its argv
and stdin, a python that imports anything, a listening socket for GraspGen-X
(CASCADE_GRASPGENX_PORT), a private HOME and ARM=mock (no can0 probe). Nothing
reaches a real port of this host.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.config import load_demo_config

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "booth_up.sh"

SAM_WARN = "[!] no SAM weights: detections without masks; run the installer"
BRAIN_WARN = "[!] no object names: start a Qwen brain or set CASCADE_PERCEPTION_URL"
HEALTH = "http://127.0.0.1:8080/health"            # check 5: the offline agent brain
DEFAULT_MODELS = "http://127.0.0.1:8080/v1/models"  # the perception default + /models

FAKE_CURL = """#!{python}
import json, os, sys
argv = sys.argv[1:]
stdin = sys.stdin.read() if "@-" in argv else None   # real curl reads `-H @-` from stdin
with open(os.environ["FAKE_CURL_LOG"], "a") as log:
    log.write(json.dumps({{"argv": argv, "stdin": stdin}}) + "\\n")
sys.exit(0 if argv[-1] in os.environ.get("FAKE_CURL_UP", "").split() else 7)
"""


def _executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def booth(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, repo / "scripts" / "booth_up.sh")
    (repo / "configs").mkdir()
    (repo / "configs" / "booth.yaml").write_text("{}\n")
    (repo / "models").mkdir()
    # Everything the other checks look for, so a red run is the new checks only.
    (repo / ".mcp.json").write_text(json.dumps({"mcpServers": {"cascade": {"env": {
        "YOLO_OFFLINE": "True", "CASCADE_BOOTH": "1", "CASCADE_ARM": "mock"}}}}, indent=1))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _executable(bin_dir / "curl", FAKE_CURL.format(python=sys.executable))
    _executable(bin_dir / "python", "#!/bin/sh\nexit 0\n")
    home = tmp_path / "home"
    home.mkdir()
    curl_log = tmp_path / "curl.jsonl"
    graspgen = socket.create_server(("127.0.0.1", 0))  # connects complete; nothing is read

    def run(up=(HEALTH, DEFAULT_MODELS), **extra):
        env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "HOME": str(home), "LANG": "C",
               "PY": str(bin_dir / "python"), "ARM": "mock",
               "CASCADE_GRASPGENX_PORT": str(graspgen.getsockname()[1]),
               "FAKE_CURL_LOG": str(curl_log), "FAKE_CURL_UP": " ".join(up)}
        env.update(extra)
        curl_log.write_text("")
        result = subprocess.run(["bash", str(repo / "scripts" / "booth_up.sh")], env=env,
                                capture_output=True, text=True, timeout=60)
        calls = [json.loads(line) for line in curl_log.read_text().splitlines()]
        return result, calls

    try:
        yield SimpleNamespace(run=run, repo=repo, home=home, sam=repo / "models" / "sam2.1_l.pt")
    finally:
        graspgen.close()


def _lines(result) -> list[str]:
    return result.stdout.splitlines()


def _perception_call(calls) -> dict:
    [call] = [c for c in calls if c["argv"][-1].endswith("/models")]
    return call


def test_the_script_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True).returncode == 0


def test_sam_weights_and_a_perception_brain_keep_the_run_green(booth):
    booth.sam.write_bytes(b"\0" * 64)
    result, _ = booth.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"[+] SAM weights: {booth.sam}" in _lines(result)
    assert "[+] perception brain answers at http://127.0.0.1:8080/v1" in _lines(result)
    assert "=== ALL CHECKS GREEN. Launch the attendee session: ===" in _lines(result)
    assert SAM_WARN not in result.stdout and BRAIN_WARN not in result.stdout


@pytest.mark.parametrize("weights", [None, b""], ids=["absent", "empty"])
def test_missing_sam_weights_warn_and_turn_the_run_red(booth, weights):
    if weights is not None:
        booth.sam.write_bytes(weights)
    result, _ = booth.run()
    assert result.returncode == 1, result.stdout
    assert any(line.startswith(SAM_WARN) for line in _lines(result)), result.stdout
    assert "[+] perception brain answers at http://127.0.0.1:8080/v1" in _lines(result)
    assert "=== FIX THE [!] LINES ABOVE, then launch: ===" in _lines(result)


def test_a_silent_perception_brain_warns_and_turns_the_run_red(booth):
    booth.sam.write_bytes(b"\0" * 64)
    result, _ = booth.run(up=(HEALTH,))
    assert result.returncode == 1, result.stdout
    assert any(line.startswith(BRAIN_WARN) for line in _lines(result)), result.stdout
    assert f"[+] SAM weights: {booth.sam}" in _lines(result)
    assert "=== FIX THE [!] LINES ABOVE, then launch: ===" in _lines(result)


def test_the_probe_asks_the_demo_yaml_default_within_two_seconds(booth, monkeypatch):
    monkeypatch.delenv("CASCADE_PERCEPTION_URL", raising=False)
    _, calls = booth.run()
    argv = _perception_call(calls)["argv"]
    assert argv[-1] == load_demo_config(camera="isaac").detector.base_url + "/models" == DEFAULT_MODELS
    assert argv[argv.index("-m") + 1] == "2"


@pytest.mark.parametrize("value,probed", [
    ("http://10.0.0.7:8000/v1", "http://10.0.0.7:8000/v1/models"),
    ("http://10.0.0.7:8000/v1/", "http://10.0.0.7:8000/v1/models"),
    ("", DEFAULT_MODELS),   # empty counts as unset, as in demo.yaml
])
def test_the_probe_follows_cascade_perception_url(booth, value, probed):
    result, calls = booth.run(up=(HEALTH, probed), CASCADE_PERCEPTION_URL=value)
    assert _perception_call(calls)["argv"][-1] == probed
    assert BRAIN_WARN not in result.stdout


def test_the_key_file_goes_on_stdin_never_in_argv(booth, tmp_path):
    key = tmp_path / "brain.key"
    key.write_text("sk-booth-7f3e\n")
    result, calls = booth.run(CASCADE_PERCEPTION_KEY_FILE=str(key))
    call = _perception_call(calls)
    assert call["argv"][call["argv"].index("-H") + 1] == "@-"
    assert call["argv"][call["argv"].index("-m") + 1] == "2"
    assert call["stdin"] == "Authorization: Bearer sk-booth-7f3e\n"
    assert all("sk-booth-7f3e" not in arg for c in calls for arg in c["argv"])
    assert "sk-booth-7f3e" not in result.stdout + result.stderr


def test_the_default_key_file_is_demo_yaml_default(booth):
    key = booth.home / ".cascade" / "nemoclaw" / "llama.key"
    key.parent.mkdir(parents=True)
    key.write_text("sk-default-91\n")
    _, calls = booth.run()
    assert _perception_call(calls)["stdin"] == "Authorization: Bearer sk-default-91\n"


def test_no_key_file_sends_no_header(booth):
    _, calls = booth.run()
    call = _perception_call(calls)
    assert "-H" not in call["argv"] and call["stdin"] is None
