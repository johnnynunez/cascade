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


def quick_path_blocks():
    return re.findall(r"```bash\n(.*?)\n```",
                      GUIDE.read_text().split("## System prerequisites")[0], re.DOTALL)


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
    commands = [part.strip() for block in bash_blocks() for part in block.split("&&")
                if part.strip().startswith("bash scripts/install.sh")]
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
    commands = [match[1] for block in quick_path_blocks()
                if (match := re.search(r"^\s*(python3 scripts/desktop.py launch.*?) 2>&1 \|$", block, re.MULTILINE))]
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


def test_documented_stop_preserves_installation_scope():
    assert any("./run.sh down\n" in block and "./run.sh down --dry-run" in block
               for block in quick_path_blocks())
    assert not any("--no-robot-turn" in block for block in bash_blocks())


@pytest.mark.parametrize("launch_exit,verify_exit", [(0, 0), (17, 0), (0, 3)])
def test_exact_start_block_bounds_output_and_preserves_launch_and_verification_failures(
    tmp_path, launch_exit, verify_exit
):
    repo = tmp_path / "paai-spark"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "desktop.py").write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "with Path('launch.calls').open('a') as log: log.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "print('[desktop] Launch progress')\n"
        "print('large-launch-detail-' * 10000)\n"
        "raise SystemExit(int(os.environ['DOC_LAUNCH_EXIT']))\n"
    )
    (scripts / "spark_verify.py").write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "Path(__file__).with_name('verify.args').write_text(json.dumps(sys.argv[1:]))\n"
        "status=int(os.environ['DOC_VERIFY_EXIT'])\n"
        "if not status: print('READY '+json.dumps({'verified':True}))\n"
        "raise SystemExit(status)\n"
    )
    block = next(block for block in quick_path_blocks() if "scripts/desktop.py launch" in block)
    assert len(block.splitlines()) <= 9, "the quick-path block must stay short"
    result = subprocess.run(["bash", "-c", block], cwd=tmp_path,
        env={"HOME": str(tmp_path), "PATH": os.defpath, "DOC_LAUNCH_EXIT": str(launch_exit),
             "DOC_VERIFY_EXIT": str(verify_exit)}, capture_output=True, text=True, timeout=10)
    assert result.returncode == (launch_exit or verify_exit), result.stderr
    assert len(result.stdout) < 1000
    assert "large-launch-detail" not in result.stdout
    assert "large-launch-detail" in (tmp_path / "paai-spark-launch.log").read_text()
    assert len((repo / "launch.calls").read_text().splitlines()) == 1
    if launch_exit:
        assert not (scripts / "verify.args").exists()
    else:
        pin = re.search(r"--ref ([0-9a-f]{40})", bootstrap_command())[1]
        assert json.loads((scripts / "verify.args").read_text()) == ["--repo", str(repo), "--expected-ref", pin]
    assert ("READY " in result.stdout) == (launch_exit == verify_exit == 0)


@pytest.mark.parametrize("stop_exit,dry_exit,leftover", [(0, 0, False), (1, 0, False),
                                                       (0, 3, False), (0, 0, True)])
def test_documented_stop_cannot_report_success_with_remaining_processes(
    tmp_path, stop_exit, dry_exit, leftover
):
    repo = tmp_path / "paai-spark"
    repo.mkdir()
    script = repo / "run.sh"
    script.write_text(
        "#!/bin/bash\n"
        'if [[ "$*" == "down --dry-run" ]]; then\n'
        '  [[ "$DOC_LEFTOVER" != 1 ]] || printf "would stop isaac pid=123\\n"\n'
        '  exit "$DOC_DRY_EXIT"\n'
        'fi\nexit "$DOC_STOP_EXIT"\n'
    )
    script.chmod(0o755)
    command = next(block for block in quick_path_blocks() if "./run.sh down\n" in block)
    result = subprocess.run(
        ["bash", "-c", command], cwd=tmp_path,
        env={"HOME": str(tmp_path), "PATH": os.defpath, "DOC_STOP_EXIT": str(stop_exit),
             "DOC_DRY_EXIT": str(dry_exit), "DOC_LEFTOVER": str(int(leftover))},
        text=True, capture_output=True, timeout=10,
    )
    if stop_exit or dry_exit or leftover:
        assert result.returncode != 0
        assert "STOPPED" not in result.stdout
    else:
        assert result.returncode == 0
        assert "STOPPED" in result.stdout


@pytest.mark.parametrize("download_exit,check_exit", [(0, 0), (22, 0), (0, 2)])
def test_documented_prerequisite_check_uses_install_pin_and_preserves_failure(
    tmp_path, download_exit, check_exit
):
    command = next(block for block in quick_path_blocks() if "spark_prerequisites.py" in block)
    match = re.search(r"/cascade/([0-9a-f]{40})/scripts/spark_prerequisites\.py", command)
    assert match
    assert f"--ref {match[1]} " in bootstrap_command()
    bins = tmp_path / "bin"
    bins.mkdir()
    curl = bins / "curl"
    curl.write_text(
        "#!/bin/bash\n"
        '[[ "$DOC_DOWNLOAD_EXIT" == 0 ]] || exit "$DOC_DOWNLOAD_EXIT"\n'
        "cat <<'PAYLOAD'\n"
        'import os\n'
        'raise SystemExit(int(os.environ["DOC_CHECK_EXIT"]))\n'
        "PAYLOAD\n"
    )
    curl.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", command], cwd=tmp_path,
        env={"HOME": str(tmp_path), "PATH": str(bins) + os.pathsep + os.defpath,
             "DOC_DOWNLOAD_EXIT": str(download_exit), "DOC_CHECK_EXIT": str(check_exit)},
        text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == (download_exit or check_exit)


def test_quick_path_registers_one_click_and_bonus_stays_optional_and_last():
    guide = GUIDE.read_text()
    assert len(quick_path_blocks()) == 6
    assert 'gio launch "$HOME/paai-spark/runs/.install/paai-spark.desktop"' in quick_path_blocks()
    assert "PAAI (Spark)" in guide
    assert "chromium-browser" in guide
    assert "Skip it when Step 1 already passes" in guide
    headings = re.findall(r"^## (.+)$", guide, re.MULTILINE)
    assert headings[-1] == "Bonus Track: Always-on demo on ngrok for testing purposes"
    bonus = guide.split("## " + headings[-1], 1)[1]
    assert "not part of the official setup" in bonus
    assert "own monitor and keyboard" in bonus
    assert all(f'spark_public.py" {action} ' in bonus for action in ("enable", "check", "disable"))
    assert not re.search(r"https?://[^\s]+\.ngrok\.", bonus)
    assert "The demo does not use port 8090" in guide


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
        "attached: false", "attached: true", "session_id", "process",
        "physics.pass: true", "physics.event_cameras.pass: true", "props_reset",
        "gpu-physical-audit.json", "Agent checklist",
    ):
        assert contract in guide
    assert re.search(r"\b(?:Apple|Mac|macOS)\b", guide, re.IGNORECASE) is None
