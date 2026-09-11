"""Front-door regressions; boundary doubles are NOT GPU/physics evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from test_launch_delivery import launcher_boundary as _launcher_boundary
from test_launch_delivery import model_http_boundary as _model_http_boundary
from test_spark_install import boundary_env, commands, executable, launch_fixture, run_stdin, support_module

launcher_boundary = _launcher_boundary
model_http_boundary = _model_http_boundary
REPO = Path(__file__).resolve().parents[1]


def test_private_delivery_artifacts_are_ignored_without_hiding_manifests(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    shutil.copy2(REPO / ".gitignore", tmp_path / ".gitignore")
    artifacts = [
        ".openclaw-cli/bin/openclaw", ".cosmos-omni/bin/python",
        ".cosmos/bin/vllm", ".isaacsim/bin/python",
        "models/Cosmos3-Edge-hf/config.json", "models/.Cosmos3-Edge-hf.lock",
        "models/.Cosmos3-Edge-hf.staging-test/model.safetensors",
    ]
    result = subprocess.run(["git", "check-ignore", "--stdin"], cwd=tmp_path,
                            input="\n".join(artifacts) + "\n", capture_output=True, text=True)
    assert set(result.stdout.splitlines()) == set(artifacts)
    visible = ["pyproject.toml", "uv.lock", "package.json", "package-lock.json",
               "models/manifest.json", "models/yoloe-11s-seg.pt", "models/yoloe-11s-seg-pf.pt"]
    result = subprocess.run(["git", "check-ignore", "--stdin"], cwd=tmp_path,
                            input="\n".join(visible) + "\n", capture_output=True, text=True)
    assert result.returncode == 1 and not result.stdout


def source_release(tmp_path, *, version="6.1.0"):
    """Run real metadata/experience validation, never SimulationApp."""
    release = tmp_path / "Isaac source release"
    (release / "apps").mkdir(parents=True)
    (release / "apps/isaacsim.exp.full.newton.kit").write_text(f'[package]\nversion="{version}"\n')
    runner = release / "metadata_runner.py"
    runner.write_text(
        "import runpy,sys\n"
        f"sys.executable={str(release / 'kit/python/bin/python3')!r}\n"
        "sys.argv=sys.argv[1:]\nrunpy.run_path(sys.argv[0],run_name='__main__')\n"
    )
    executable(release / "python.sh", f'#!/bin/bash\nexec "{sys.executable}" "{runner}" "$@"\n')
    return release


@pytest.mark.parametrize("check", [True, False])
def test_source_isaac_install_reuses_release_without_any_package_install(tmp_path, check):
    env, log = boundary_env(tmp_path)
    release = source_release(tmp_path)
    env["ISAACSIM_PATH"] = str(release)
    repo = Path(env["BOUNDARY_SOURCE"])
    shutil.copy2(REPO / "scripts/isaac_runtime.py", repo / "scripts/isaac_runtime.py")
    args = ["--dir", str(repo), "--check" if check else "--accept-eula"]
    result = run_stdin(tmp_path, *args, entry="install_isaac.sh", env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"layout": "source"' in result.stdout, result.stdout
    assert not any(c[0] == "uv" for c in commands(log)), "must NEVER pip-install into Kit Python"


@pytest.mark.parametrize("entry", ["install", "launch"])
def test_source_metadata_preflight_strips_inherited_python_and_libraries(launcher_boundary, tmp_path, entry):
    h = launcher_boundary
    release = source_release(tmp_path)
    wrapper = release / "python.sh"
    # Match python.sh's override hazard without ever invoking Kit or pip.
    wrapper.write_text(wrapper.read_text().replace(
        "#!/bin/bash\n", "#!/bin/bash\n"
        'for key in PYTHONEXE PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX LD_LIBRARY_PATH LD_PRELOAD; do\n'
        '  [[ -z "${!key:-}" ]] || { printf "inherited %s\\n" "$key" >&2; exit 41; }\n'
        'done\n'
    ))
    env = dict(h["env"], ISAACSIM_PATH=str(release), PYTHONEXE="/agent/python",
               VIRTUAL_ENV="/agent", LD_LIBRARY_PATH="/agent/lib", PYTHONPATH="/agent/src")
    env.pop("ISAACSIM_PYTHON_EXE", None)
    if entry == "install":
        command = ["bash", str(REPO / "scripts/install_isaac.sh"), "--dir", str(h["repo"]), "--check"]
    else:
        command = list(h["command"]) + ["--check"]
        command[command.index("--sim") + 1] = "isaac"
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert '"layout": "source"' in result.stdout, result.stdout + result.stderr
    assert "inherited " not in result.stdout + result.stderr
    assert not (h["repo"] / ".isaacsim").exists()


def test_explicit_bad_source_fails_closed_instead_of_installing_wheel(tmp_path):
    env, log = boundary_env(tmp_path)
    env["ISAACSIM_PATH"] = str(source_release(tmp_path, version="6.0.0"))
    repo = Path(env["BOUNDARY_SOURCE"])
    shutil.copy2(REPO / "scripts/isaac_runtime.py", repo / "scripts/isaac_runtime.py")
    result = run_stdin(tmp_path, "--dir", str(repo), "--accept-eula", entry="install_isaac.sh", env=env)
    assert result.returncode != 0
    assert "declares" in result.stdout + result.stderr
    assert not any(c[0] == "uv" for c in commands(log))


def test_source_selection_survives_helper_launch(tmp_path, monkeypatch):
    helper, repo = launch_fixture(tmp_path, monkeypatch)
    source = source_release(tmp_path)
    monkeypatch.setenv("ISAACSIM_PATH", str(source))
    monkeypatch.delenv("ISAACSIM_PYTHON_EXE", raising=False)
    assert helper.launch(repo, "spark", "cosmos", no_open=True) == 0
    try:
        record = json.loads((repo / "launch-record.json").read_text())
        assert record["env"]["ISAACSIM_PYTHON_EXE"] == str(source / "python.sh")
    finally:
        for process in helper._fixture_cosmos:
            helper.stop_group(process)


def test_direct_helper_launch_does_not_grant_eula_consent(tmp_path, monkeypatch):
    helper = support_module()
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    monkeypatch.setattr(helper, "model_health", lambda: True)
    monkeypatch.setattr(helper.subprocess, "Popen", lambda *a, **kw: pytest.fail("spawned without consent"))
    with pytest.raises(RuntimeError, match="consent|accept-eula"):
        helper.launch(tmp_path, "spark", "cosmos")
    assert not (tmp_path / "runs/.launch").exists()


def test_spark_cannot_keep_an_unproven_cloud_brain(tmp_path):
    result = run_stdin(tmp_path, "--profile", "spark", "--brain", "keep", "--dry-run")
    assert result.returncode != 0
    assert "cosmos" in result.stderr.lower()


def test_install_check_accepts_private_npm_cli_layout(tmp_path, monkeypatch):
    helper = support_module()
    package = tmp_path / ".openclaw-cli/lib/node_modules/openclaw/package.json"
    package.parent.mkdir(parents=True)
    package.write_text('{"version":"2026.9.3"}')
    monkeypatch.setattr(helper, "scene_problems", lambda repo: [])
    monkeypatch.setattr(helper, "model_asset_valid", lambda *args: True)
    monkeypatch.setattr(helper.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))
    problems = helper.installation_problems(tmp_path, "spark", "cosmos")
    assert not any("OpenClaw" in p for p in problems), problems


def test_launch_failure_before_proof_invalidates_previous_green_receipt(launcher_boundary):
    h = launcher_boundary
    state = Path(h["env"]["CASCADE_LAUNCH_STATE"]) / "profile-isolated-test"
    state.mkdir(parents=True)
    (state / "proof.json").write_text('{"verified":true,"session_id":"old"}')
    # Fail after dependency checks but before the proof runner is invoked.
    h["env"]["CASCADE_DETECTOR_MODEL"] = str(h["repo"] / "missing.pt")
    result = subprocess.run(h["command"], env=h["env"], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert json.loads((state / "proof.json").read_text())["verified"] is False


def test_supervised_cosmos_startup_failure_invalidates_previous_green(tmp_path, monkeypatch):
    helper, repo = launch_fixture(tmp_path, monkeypatch)
    state = repo / "runs/.launch/profile-cascade-demo"
    state.mkdir(parents=True)
    previous = '{"verified":true,"session_id":"previous-attendee"}'
    (state / "proof.json").write_text(previous)

    def unhealthy_model():
        raise RuntimeError("test boundary: wrong model on the selected endpoint")

    monkeypatch.setattr(helper, "model_health", unhealthy_model)
    with pytest.raises(RuntimeError, match="wrong model"):
        helper.launch(repo, "spark", "cosmos")
    current = json.loads((state / "proof.json").read_text())
    assert current["verified"] is False
    assert current["profile"] == "cascade-demo"
    assert (state / current["attempt"] / "previous-proof.json").read_text() == previous
    assert not (repo / "launch-record.json").exists()
    assert helper._fixture_cosmos == []


def test_launch_preserves_profile_home_workspace_and_open_vocabulary(launcher_boundary):
    h = launcher_boundary
    h["env"].pop("CASCADE_DETECT_CLASSES", None)
    result = subprocess.run(h["command"], env=h["env"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in Path(h["env"]["HOST_LOG"]).read_text().splitlines()]
    workspace = str(Path(h["env"]["HOST_CFG"]).parent / "workspace")
    for key in ("agents.defaults.workspace", "agents.entries.main.workspace"):
        updates = [c for c in calls if key in c]
        assert updates and updates[-1][updates[-1].index(key) + 1] == workspace, updates
    config = json.loads(Path(h["env"]["MCP_CONFIG"]).read_text())
    assert "CASCADE_DETECT_CLASSES" not in config["env"], "default must be prompt-free, not a fixed scene vocabulary"


def test_mcp_uses_profile_owned_memories_unless_operator_overrides(launcher_boundary):
    h = launcher_boundary
    h["env"].pop("CASCADE_GRASP_MEMORY_PATH", None)
    h["env"].pop("CASCADE_BELIEFS_PATH", None)
    envelope = str(h["repo"] / "operator-envelope.json")
    h["env"]["CASCADE_ENVELOPE_PATH"] = envelope
    result = subprocess.run(h["command"], env=h["env"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    config = json.loads(Path(h["env"]["MCP_CONFIG"]).read_text())
    state = Path(h["env"]["CASCADE_LAUNCH_STATE"]) / "profile-isolated-test"
    assert config["env"].get("CASCADE_GRASP_MEMORY_PATH") == str(state / "memory/grasp_memory.json")
    assert config["env"].get("CASCADE_BELIEFS_PATH") == str(state / "memory/beliefs.json")
    assert config["env"].get("CASCADE_ENVELOPE_PATH") == envelope


def test_run_restores_installed_source_profile_and_forces_spark_contract(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(REPO / "run.sh", repo / "run.sh")
    state = repo / "runs/.install"
    state.mkdir(parents=True)
    (state / "env.sh").write_text(
        "export CASCADE_INSTALL_PROFILE=spark\n"
        "export CASCADE_OPENCLAW_PROFILE=cascade-demo\n"
        "export ISAACSIM_PATH='/a/source release'\n"
    )
    executable(repo / "scripts/launch.sh", '#!/bin/bash\nprintf "%s\\n" "$CASCADE_OPENCLAW_PROFILE" "$ISAACSIM_PATH" "$@"\n')
    result = subprocess.run(["bash", str(repo / "run.sh"), "--dry-run"], env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[:2] == ["cascade-demo", "/a/source release"]
    assert "--setup" not in lines
    assert lines[lines.index("--sim") + 1] == "isaac"
    assert lines[lines.index("--brain") + 1] == "cosmos"


@pytest.mark.parametrize("args", [["--sim"], ["--occupancy", "bogus"], ["--graspgenx", "bogus"]])
def test_bad_launch_arguments_fail_before_writes(tmp_path, args):
    state = tmp_path / "state"
    result = subprocess.run(["bash", str(REPO / "scripts/launch.sh"), "--dry-run", *args], env={**os.environ, "CASCADE_LAUNCH_STATE": str(state)}, capture_output=True, text=True, timeout=20)
    assert result.returncode == 2, result.stdout + result.stderr
    assert not state.exists()


def test_spark_installer_registers_desktop_entries_after_preparation(tmp_path):
    env, log = boundary_env(tmp_path)
    result = run_stdin(tmp_path, "--accept-eula", "--prepare-only", env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = commands(log)
    assert any("register" in c and any(x.endswith("desktop.py") for x in c) for c in calls)


@pytest.mark.parametrize("prepare_only", [False, True])
def test_installer_registration_creates_real_entries_only_in_private_home(tmp_path, prepare_only):
    env, log = boundary_env(tmp_path)
    env.update(BOUNDARY_REAL_DESKTOP="1", XDG_DATA_HOME=str(tmp_path / "data"),
               XDG_CONFIG_HOME=str(tmp_path / "home/.config"))
    desktop = tmp_path / "home/Desktop"
    desktop.mkdir(parents=True)
    config = tmp_path / "home/.config"
    config.mkdir()
    (config / "user-dirs.dirs").write_text('XDG_DESKTOP_DIR="$HOME/Desktop"\n')
    apps = tmp_path / "data/applications"
    apps.mkdir(parents=True)
    unrelated = apps / "unrelated.desktop"
    unrelated.write_text("operator's existing shortcut")
    result = run_stdin(tmp_path, "--accept-eula", *(["--prepare-only"] if prepare_only else []), env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = commands(log)
    recording = next(i for i, c in enumerate(calls) if "record" in c)
    registration = [i for i, c in enumerate(calls) if "register" in c and any(x.endswith("desktop.py") for x in c)]
    assert len(registration) == 1 and registration[0] > recording
    launches = [i for i, c in enumerate(calls) if "launch" in c]
    assert not launches if prepare_only else launches[0] > registration[0]
    for directory in (apps, desktop):
        entries = sorted(directory.glob("cascade-*.desktop"))
        assert len(entries) == 2
        assert {next(line for line in p.read_text().splitlines() if line.startswith("Name=")) for p in entries} == {
            "Name=CASCADE (Spark)", "Name=Install CASCADE (Spark)"}
        assert all("--accept-eula" not in p.read_text() and os.access(p, os.X_OK) for p in entries)
    assert unrelated.read_text() == "operator's existing shortcut"
    assert not (tmp_path / "home/cascade/runs/.launch").exists()


def test_spark_launcher_never_repairs_missing_deps_with_unpinned_pip(launcher_boundary):
    h = launcher_boundary
    h["env"]["CASCADE_INSTALL_PROFILE"] = "spark"
    h["env"]["HOME"] = str(h["repo"] / "private-home")
    h["env"]["ISAACSIM_PYTHON_EXE"] = sys.executable  # metadata boundary, never reached
    command = list(h["command"])
    command[command.index("--sim") + 1] = "isaac"
    wrapper = Path(h["env"]["PY"])
    wrapper.write_text(wrapper.read_text().replace("print('')  # dependency boundary", "print('missing-perception')  # dependency boundary"))
    attempted = h["repo"] / "unexpected-install"
    executable(wrapper.parent / "uv", f'#!/bin/bash\nprintf invoked > "{attempted}"\nexit 97\n')
    result = subprocess.run(command, env=h["env"], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert not attempted.exists(), "Spark launch must not install arbitrary wheels into the selected app/Kit Python"
    assert "install.sh" in result.stdout + result.stderr


def test_mcp_preserves_explicit_view_disable(launcher_boundary):
    h = launcher_boundary
    h["env"]["CASCADE_VIEW"] = "0"
    wrapper = Path(h["env"]["PY"])
    wrapper.write_text(wrapper.read_text().replace("if args==['-c','import torch']:", "if args in (['-c','import torch'], ['-c','import mujoco']):"))
    command = list(h["command"])
    command[command.index("--sim") + 1] = "mujoco"
    result = subprocess.run(command, env=h["env"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    config = json.loads(Path(h["env"]["MCP_CONFIG"]).read_text())
    assert config["env"].get("CASCADE_VIEW") == "0"


def test_isaac_cold_start_budget_is_visible_and_at_least_twenty_minutes(tmp_path):
    env = {**os.environ, "CASCADE_LAUNCH_STATE": str(tmp_path / "state")}
    env.pop("ISAAC_WAIT_S", None)
    result = subprocess.run(["bash", str(REPO / "scripts/launch.sh"), "--dry-run", "--sim", "isaac", "--brain", "cosmos"],
                            env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0
    assert "startup budget=1200s" in result.stdout


def test_reused_isaac_bridge_is_probed_not_accepted_on_port_alone(launcher_boundary, monkeypatch):
    from test_isaac_bridge import bridge_port as bridge_fixture

    bridge = bridge_fixture.__wrapped__()
    port, _ = next(bridge)
    h = launcher_boundary
    isaac = h["repo"] / "python.sh"
    executable(isaac, '#!/bin/bash\nprintf \'{"layout":"test-double","version":"6.1.0"}\\n\'\n')
    usd = h["repo"] / "scene.usda"
    usd.write_text("# metadata placeholder: never loaded")
    h["env"].update(ISAACSIM_PYTHON_EXE=str(isaac), CASCADE_BRIDGE_PORT=str(port), CASCADE_USD=str(usd))
    command = list(h["command"])
    command[command.index("--sim") + 1] = "isaac"
    try:
        result = subprocess.run(command, env=h["env"], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Isaac bridge answers:" in result.stdout
    finally:
        try:
            next(bridge)
        except StopIteration:
            pass
