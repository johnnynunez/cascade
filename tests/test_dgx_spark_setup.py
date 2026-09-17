"""Exercise the published setup commands without downloads or live services."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/DGX_SPARK_SETUP.md"


def bash_blocks():
    return re.findall(r"```bash\n(.*?)\n```", GUIDE.read_text(), re.DOTALL)


def bootstrap_command():
    commands = [block for block in bash_blocks() if "/scripts/bootstrap.sh" in block]
    assert len(commands) == 1, "publish one canonical bootstrap command"
    return commands[0]


def test_setup_shell_examples_are_valid_bash():
    for block in bash_blocks():
        result = subprocess.run(
            ["bash", "-n"], input=block, text=True, capture_output=True, timeout=10
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("download_exit,install_exit", [(0, 0), (22, 0), (0, 2), (0, 17)])
def test_bootstrap_preserves_failures_and_explicit_preparation(
    tmp_path, download_exit, install_exit
):
    """Run the exact public pipeline with only its network boundary replaced."""
    command = bootstrap_command()
    match = re.search(
        r"https://raw\.githubusercontent\.com/johnnynunez/cascade/"
        r"([0-9a-f]{40})/scripts/bootstrap\.sh",
        command,
    )
    assert match, "bootstrap source must be an immutable commit"
    bins = tmp_path / "bin"
    bins.mkdir()
    curl = bins / "curl"
    curl.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$@" > "$HOME/curl.args"\n'
        '[[ "$DOC_DOWNLOAD_EXIT" == 0 ]] || exit "$DOC_DOWNLOAD_EXIT"\n'
        "cat <<'PAYLOAD'\n"
        'printf "%s\\n" "$@" > "$HOME/bootstrap.args"\n'
        'printf "installer stdout\\n"\n'
        'printf "installer diagnostic\\n" >&2\n'
        'exit "$DOC_INSTALL_EXIT"\n'
        "PAYLOAD\n"
    )
    curl.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", command], cwd=tmp_path,
        env={
            "HOME": str(tmp_path), "PATH": str(bins) + os.pathsep + os.defpath,
            "DOC_DOWNLOAD_EXIT": str(download_exit),
            "DOC_INSTALL_EXIT": str(install_exit),
        },
        text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == (download_exit or install_exit), result.stderr
    assert match[0] in (tmp_path / "curl.args").read_text().splitlines()
    if not download_exit:
        assert (tmp_path / "bootstrap.args").read_text().splitlines() == [
            "--ref", match[1], "--profile", "spark", "--accept-eula",
            "--prepare-only", "--dir", str(tmp_path / "paai-spark"),
        ]
        log = (tmp_path / "paai-spark-install.log").read_text()
        assert "installer stdout" in log
        assert "installer diagnostic" in log


def test_preparation_check_does_not_accept_eula_or_start_services():
    commands = [line for block in bash_blocks() for line in block.splitlines()
                if line.startswith("bash scripts/install.sh")]
    assert len(commands) == 1
    assert shlex.split(commands[0]) == [
        "bash", "scripts/install.sh", "--profile", "spark", "--dir", "$PWD", "--check",
    ]
    guide = GUIDE.read_text()
    assert "https://docs.omniverse.nvidia.com/eula" in guide
    assert "OMNI_KIT_ACCEPT_EULA=YES" in guide
    assert "alone does not grant consent" in guide
    assert "runs/.install/install.json" in guide


def test_documented_launch_uses_the_desktop_supervisor(tmp_path):
    commands = [block for block in bash_blocks()
                if block.startswith("python3 scripts/desktop.py launch")]
    assert len(commands) == 1
    args = shlex.split(commands[0])
    assert args == ["python3", "scripts/desktop.py", "launch", "--repo", "$PWD",
                    "--headless", "--no-open"]
    result = subprocess.run(
        [sys.executable, str(ROOT / args[1]), *[
            str(tmp_path) if arg == "$PWD" else arg for arg in args[2:]
        ], "--dry-run"],
        text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["services_started"] is False
    assert plan["command_after_consent"] == [
        str(tmp_path / ".venv/bin/python"), str(tmp_path / "scripts/install_support.py"),
        "launch", "--repo", str(tmp_path), "--profile", "spark", "--brain", "qwen",
        "--headless", "--no-open",
    ]


def test_openclaw_commands_and_stop_preserve_installation_scope():
    commands = [block.replace("\\\n", "") for block in bash_blocks()
                if ".openclaw-cli/bin/openclaw" in block]
    assert len(commands) == 2
    for block in commands:
        tokens = shlex.split(block.splitlines()[0])
        assert tokens[:4] == [
            "OPENCLAW_STATE_DIR=$PWD/runs/.launch/profile-cascade-demo/openclaw",
            ".openclaw-cli/bin/openclaw", "--profile", "cascade-demo",
        ]
        assert tokens[4:] in (["health", "--json"], ["dashboard"])
    assert "./run.sh down\n./run.sh down --dry-run" in bash_blocks()
    assert not any("--no-robot-turn" in block for block in bash_blocks())


def test_natural_examples_match_current_startup_proof():
    guide_prompts = re.findall(r"^> (.+)$", GUIDE.read_text(), re.MULTILINE)
    tree = ast.parse((ROOT / "scripts/demo_proof.py").read_text())
    proof = next(node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "_run_spark_cases")
    messages = {node.value for node in ast.walk(proof)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert len(guide_prompts) == 4
    assert all(prompt in messages for prompt in guide_prompts)


def test_ready_requires_current_machine_readable_evidence():
    guide = GUIDE.read_text()
    for contract in (
        "PREPARED", "READY", "STARTED (UNVERIFIED: robot proof skipped)",
        "runs/.install/desktop-latest.json", "runs/.launch/profile-cascade-demo/proof.json",
        "exit_code: 0", "verified: true", "started_at", "finished_at",
        "physics.pass: true", "physics.event_cameras.pass: true", "props_reset",
        "gpu-physical-audit.json", "Agent checklist",
    ):
        assert contract in guide
    assert re.search(r"\b(?:Apple|Mac|macOS)\b", guide, re.IGNORECASE) is None
