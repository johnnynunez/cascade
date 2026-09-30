"""Installer failure/retry boundaries; no GPU packages or public downloads."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from test_spark_install import executable, support_module


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def installer(tmp_path):
    repo = tmp_path / "cascade"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "source.txt").write_text("pinned fixture\n")
    subprocess.run(["git", "init", "-q", str(upstream)], check=True)
    subprocess.run(["git", "-C", str(upstream), "add", "."], check=True)
    subprocess.run(["git", "-C", str(upstream), "-c", "user.name=Fixture", "-c",
                    "user.email=fixture@example.invalid", "commit", "-qm", "Fixture"], check=True)
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    source = (ROOT / "scripts/install_graspgenx.sh").read_text()
    source = re.sub(r"(?m)^(SOURCE_REF|CHECKPOINT_REF|GRIPPER_REF)=\w+$",
                    lambda match: f"{match[1]}={revision}", source)
    (scripts / "install_graspgenx.sh").write_text(source)
    shutil.copy2(ROOT / "scripts/requirements-graspgenx.txt", scripts)
    bins = tmp_path / "bin"
    log = tmp_path / "calls.jsonl"
    prefix = f'#!{sys.executable}\n' + """
import json, os, pathlib, subprocess, sys
a = sys.argv[1:]
with open(os.environ['GGX_LOG'], 'a') as stream:
    stream.write(json.dumps([pathlib.Path(sys.argv[0]).name, *a]) + '\\n')
"""
    executable(bins / "git", prefix + """
mode = os.environ.get('GGX_FAIL', '')
if (mode == 'fetch' and 'fetch' in a) or (mode == 'lfs' and 'lfs' in a and 'pull' in a):
    sys.exit(55)
if 'lfs' in a and os.environ.get('GGX_REAL_LFS') != '1':
    if 'version' in a: print('git-lfs/fixture')
    if '--json' in a: print('{"files": []}')
    sys.exit(0)
a = [os.environ['GGX_UPSTREAM'] if x.startswith('https://') else x for x in a]
sys.exit(subprocess.call([os.environ['GGX_REAL_GIT'], *a]))
""")
    executable(bins / "uv", prefix + """
if a[0] == 'venv':
    target = pathlib.Path(a[-1]) / 'bin/python'
    target.parent.mkdir(parents=True)
    target.write_text('#!/bin/sh\\nexit 0\\n')
    target.chmod(0o755)
""")
    executable(bins / "sleep", "#!/bin/sh\nexit 0\n")
    env = dict(os.environ, HOME=str(tmp_path / "home"), PATH=f"{bins}:/usr/bin:/bin",
               GGX_UPSTREAM=str(upstream), GGX_REAL_GIT=shutil.which("git"), GGX_LOG=str(log))
    return repo, env, log


@pytest.mark.parametrize("failure", ["fetch", "lfs"])
def test_interrupted_source_download_does_not_poison_the_next_install(installer, failure):
    repo, env, log = installer
    command = ["bash", str(repo / "scripts/install_graspgenx.sh")]
    failed = subprocess.run(command, env=dict(env, GGX_FAIL=failure), text=True, capture_output=True)
    assert failed.returncode == 55, failed.stderr
    assert not (repo / ".graspgenx-src").exists()
    assert not list(repo.glob(".graspgenx-clone.*"))
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert sum(failure in call and (failure != "lfs" or "pull" in call) for call in calls) == 3
    assert not any(call[0] == "uv" for call in calls)

    retry = subprocess.run(command, env=env, text=True, capture_output=True)
    assert retry.returncode == 0, retry.stderr
    assert (repo / ".graspgenx-src/.git/HEAD").is_file()
    assert (repo / ".graspgenx-src/ext/graspgenx_checkpoints/.git/HEAD").is_file()
    assert (repo / ".graspgenx-src/ext/gripper_descriptions/.git/HEAD").is_file()
    assert (repo / ".graspgenx/bin/python").is_file()


def test_graspgenx_check_preserves_existing_sources_and_downloads_nothing(installer):
    repo, env, log = installer
    command = ["bash", str(repo / "scripts/install_graspgenx.sh")]
    installed = subprocess.run(command, env=env, text=True, capture_output=True)
    assert installed.returncode == 0, installed.stderr
    log.write_text("")
    check = subprocess.run([*command, "--check"], env=env, text=True, capture_output=True)
    assert check.returncode == 0, check.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert not any(call[0] == "uv" or "fetch" in call or "pull" in call for call in calls)
    source = repo / ".graspgenx-src/source.txt"
    source.write_text("operator changes\n")
    check = subprocess.run([*command, "--check"], env=env, text=True, capture_output=True)
    assert check.returncode != 0
    assert "modified source" in check.stderr
    assert source.read_text() == "operator changes\n"


def test_upstream_pointer_without_attributes_hydrates_without_staging_changes(installer):
    repo, env, log = installer
    if shutil.which("git-lfs") is None:
        pytest.skip("real Git LFS is required for the upstream packaging regression")
    upstream = Path(env["GGX_UPSTREAM"])
    payload = b"real upstream mesh payload\n"
    oid = hashlib.sha256(payload).hexdigest()
    # This is the published gripper layout bug: a committed LFS pointer whose
    # extension has no filter in .gitattributes. Include a literal glob in its
    # name to ensure generated local attributes cannot match another file.
    relative = "meshes/part [left].obj"
    pointer = upstream / relative
    pointer.parent.mkdir()
    pointer.write_text(f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {len(payload)}\n")
    (upstream / ".gitattributes").write_text("*.npy filter=lfs diff=lfs merge=lfs -text\n")
    (upstream / "meshes/part l.obj").write_text("ordinary non-LFS mesh\n")
    blob = upstream / ".git/lfs/objects" / oid[:2] / oid[2:4] / oid
    blob.parent.mkdir(parents=True)
    blob.write_bytes(payload)
    subprocess.run(["git", "-C", str(upstream), "add", "."], check=True)
    subprocess.run(["git", "-C", str(upstream), "-c", "user.name=Fixture", "-c",
                    "user.email=fixture@example.invalid", "commit", "-qm", "Incomplete upstream attributes"], check=True)
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    script = repo / "scripts/install_graspgenx.sh"
    script.write_text(re.sub(r"(?m)^(SOURCE_REF|CHECKPOINT_REF|GRIPPER_REF)=\w+$",
                            lambda match: f"{match[1]}={revision}", script.read_text()))
    env = dict(env, GGX_REAL_LFS="1")
    command = ["bash", str(script)]
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    checkout = repo / ".graspgenx-src"
    assert (checkout / relative).read_bytes() == payload
    assert (checkout / ".gitattributes").read_bytes() == (upstream / ".gitattributes").read_bytes()
    assert not subprocess.check_output(["git", "--no-optional-locks", "-C", str(checkout),
                                       "status", "--porcelain", "--untracked-files=no"])
    assert "unspecified" in subprocess.check_output(
        ["git", "-C", str(checkout), "check-attr", "filter", "--", "meshes/part l.obj"], text=True)
    (checkout / relative).write_bytes(b"operator mesh edits\n")
    failed = subprocess.run(command, env=env, text=True, capture_output=True)
    assert failed.returncode != 0
    assert "modified source" in failed.stderr
    assert (checkout / relative).read_bytes() == b"operator mesh edits\n"


@pytest.mark.parametrize("failure", ["LFS pointer", "unexpected revision", "torch version mismatch"])
def test_spark_preflight_propagates_graspgenx_identity_failure(tmp_path, monkeypatch, failure):
    support = support_module()
    for relative in (".venv/bin/python", ".graspgenx/bin/python",
                     ".graspgenx-src/graspgenx/serving/zmq_server.py",
                     ".graspgenx-src/ext/graspgenx_checkpoints/release/gen/epoch_736.pth",
                     ".graspgenx-src/ext/graspgenx_checkpoints/release/dis/epoch_1056.pth"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    package = tmp_path / ".openclaw-cli/lib/node_modules/openclaw/package.json"
    package.parent.mkdir(parents=True)
    package.write_text('{"version":"2026.9.3"}')
    monkeypatch.setattr(support, "MODEL_ASSETS", {})
    monkeypatch.setattr(support, "scene_problems", lambda repo: [])
    monkeypatch.setattr(support, "kitchen_problems", lambda repo: [])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        grasp_check = any(value.endswith("/install_graspgenx.sh") for value in command)
        if grasp_check:
            assert command[-1] == "--check"
        return subprocess.CompletedProcess(command, int(grasp_check), stdout="",
                                           stderr=failure if grasp_check else "")

    monkeypatch.setattr(support.subprocess, "run", run)
    assert support.installation_problems(tmp_path, "spark", "qwen") == [f"GraspGen-X preflight:  {failure}"]
    assert any(value.endswith("/install_graspgenx.sh") for call in calls for value in call)
