"""Linux desktop entry + real subprocess controller tests; no GPU services."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]


def controller():
    path = ROOT / "scripts/desktop.py"
    assert path.is_file(), "missing local desktop controller"
    spec = importlib.util.spec_from_file_location("cascade_desktop", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_registration_is_idempotent_and_launches_no_services(tmp_path, monkeypatch):
    module = controller()
    repo = tmp_path / "checkout with spaces"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/desktop.py", repo / "scripts/desktop.py")
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    first = module.register(repo, desktop_dir=desktop)
    before = {str(p): p.stat().st_mtime_ns for p in first}
    assert module.register(repo, desktop_dir=desktop) == first
    assert {str(p): p.stat().st_mtime_ns for p in first} == before
    assert len(first) == 4
    for entry in first:
        text = entry.read_text()
        assert "Terminal=true" in text
        assert str(repo) in text
        assert "--accept-eula" not in text, "a desktop shortcut cannot grant consent"
        assert os.access(entry, os.X_OK)
        validate = subprocess.run(["desktop-file-validate", str(entry)], capture_output=True, text=True)
        assert validate.returncode == 0, validate.stdout + validate.stderr
    assert not (repo / "runs").exists()


def test_declined_install_does_not_write_or_spawn(tmp_path, monkeypatch):
    module = controller()
    monkeypatch.setattr(module, "confirm_eula", lambda: False)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: pytest.fail("started after declined EULA"))
    assert module.perform(tmp_path, "install") != 0
    assert not (tmp_path / "runs").exists()


def test_launch_without_consent_receipt_never_installs_or_accepts_licenses(tmp_path, monkeypatch):
    module = controller()
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: pytest.fail("started without recorded consent"))
    with pytest.raises(RuntimeError, match="Install CASCADE|consent"):
        module.perform(tmp_path, "launch")


@pytest.mark.parametrize("exit_code", [0, 17])
def test_installer_runs_existing_pipeline_and_keeps_progress_log(tmp_path, monkeypatch, capsys, exit_code):
    module = controller()
    repo = tmp_path / "repo"
    script = repo / "scripts/install.sh"
    script.parent.mkdir(parents=True)
    # Only the package/service boundary is doubled, with a real subprocess.
    script.write_text(f'#!/bin/bash\nprintf "INSTALL_STEP_ONE\\n"\nprintf "%s\\n" "$@"\nexit {exit_code}\n')
    monkeypatch.setattr(module, "confirm_eula", lambda: True)
    assert module.perform(repo, "install", prepare_only=True) == exit_code
    receipt = json.loads((repo / "runs/.install/desktop-latest.json").read_text())
    log = Path(receipt["log"])
    assert log.is_file()
    assert "INSTALL_STEP_ONE" in log.read_text()
    assert "--prepare-only" in log.read_text()
    assert "--accept-eula" in log.read_text()
    assert receipt["exit_code"] == exit_code
    assert receipt["action"] == "install"
    assert "INSTALL_STEP_ONE" in capsys.readouterr().out
    assert "READY" not in log.read_text()


def test_launch_uses_installer_supervisor_with_pinned_source_and_profile(tmp_path):
    module = controller()
    repo = tmp_path / "repo"
    script = repo / "scripts/install_support.py"
    script.parent.mkdir(parents=True)
    python = repo / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    script.write_text('import json,os,sys; print(json.dumps({"args":sys.argv[1:],"profile":os.environ.get("CASCADE_OPENCLAW_PROFILE"),"source":os.environ.get("ISAACSIM_PATH")}))\n')
    consent = repo / "runs/.install/install.json"
    consent.parent.mkdir(parents=True)
    consent.write_text(json.dumps({"repo": str(repo.resolve()), "eula_accepted": True,
                                   "eula_url": module.EULA_URL, "profile": "spark", "brain": "cosmos",
                                   "isaac_environment": {"ISAACSIM_PATH": "/selected/source", "ISAACSIM_PYTHON_EXE": "/selected/source/python.sh"}}))
    assert module.perform(repo, "launch") == 0
    latest = json.loads((repo / "runs/.install/desktop-latest.json").read_text())
    output = [json.loads(s) for s in Path(latest["log"]).read_text().splitlines() if s.startswith('{')][0]
    assert output["args"] == ["launch", "--repo", str(repo), "--profile", "spark", "--brain", "cosmos"]
    assert output["source"] == "/selected/source"
    assert output["profile"] == "cascade-demo"


def test_duplicate_click_does_not_launch_second_subprocess(tmp_path, monkeypatch):
    import fcntl

    module = controller()
    repo = tmp_path / "repo"
    state = repo / "runs/.install"
    state.mkdir(parents=True)
    monkeypatch.setattr(module, "confirm_eula", lambda: True)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: pytest.fail("duplicate operation was started"))
    with (state / "desktop.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert module.perform(repo, "install") == 75


def test_spawn_failure_keeps_failed_status_and_diagnostic_log(tmp_path, monkeypatch):
    module = controller()
    monkeypatch.setattr(module, "confirm_eula", lambda: True)

    def fail_to_spawn(*args, **kwargs):
        raise PermissionError("test boundary: process execution denied")

    monkeypatch.setattr(module.subprocess, "Popen", fail_to_spawn)
    with pytest.raises(PermissionError, match="execution denied"):
        module.perform(tmp_path, "install")
    report = json.loads((tmp_path / "runs/.install/desktop-latest.json").read_text())
    assert report["exit_code"] == 1
    assert report["finished_at"] >= report["started_at"]
    assert "execution denied" in Path(report["log"]).read_text()


@pytest.mark.skipif(not Path("/proc/self/cgroup").is_file(), reason="Linux cgroup inheritance")
def test_desktop_child_retains_the_callers_service_cgroup(tmp_path, monkeypatch):
    import shlex

    module = controller()
    script = tmp_path / "scripts/install.sh"
    script.parent.mkdir()
    # Session/process-group isolation must not escape an operator's systemd
    # memory budget. This child only reads its kernel membership, no service.
    probe = 'from pathlib import Path; print(Path("/proc/self/cgroup").read_text(), end="")'
    script.write_text(f"#!/bin/bash\nexec {shlex.quote(sys.executable)} -c {shlex.quote(probe)}\n")
    monkeypatch.setattr(module, "confirm_eula", lambda: True)
    expected = Path("/proc/self/cgroup").read_text()
    assert module.perform(tmp_path, "install", prepare_only=True) == 0
    report = json.loads((tmp_path / "runs/.install/desktop-latest.json").read_text())
    assert Path(report["log"]).read_text() == expected


def test_desktop_dry_run_does_not_prompt_write_or_spawn(tmp_path, monkeypatch):
    module = controller()
    monkeypatch.setattr(module, "confirm_eula", lambda: pytest.fail("dry run asked for consent"))
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: pytest.fail("dry run spawned"))
    assert module.perform(tmp_path, "install", dry_run=True) == 0
    assert module.perform(tmp_path, "launch", dry_run=True) == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_terminal_close_stops_only_its_child_and_records_interruption(tmp_path, signum):
    module = controller()
    repo = tmp_path / "repo"
    script = repo / "scripts/install_support.py"
    script.parent.mkdir(parents=True)
    python = repo / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    # A sleeping subprocess, NOT an installer, gateway or simulator.
    script.write_text(
        "import os,pathlib,signal,sys,time\n"
        "def stopped(*args):\n"
        "    pathlib.Path('child-stopped').touch()\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stopped)\n"
        "pathlib.Path('child.pid').write_text(str(os.getpid()))\n"
        "print('CHILD_READY', flush=True)\n"
        "time.sleep(60)\n"
    )
    consent = repo / "runs/.install/install.json"
    consent.parent.mkdir(parents=True)
    consent.write_text(json.dumps({"repo": str(repo), "eula_accepted": True,
                                  "eula_url": module.EULA_URL, "profile": "spark", "brain": "cosmos"}))
    process = subprocess.Popen([sys.executable, "-B", str(ROOT / "scripts/desktop.py"), "launch", "--repo", str(repo)],
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        marker = repo / "child.pid"
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.02)
        assert marker.is_file()
        process.send_signal(signum)
        output, _ = process.communicate(timeout=15)
        assert process.returncode == 130, output
        assert (repo / "child-stopped").exists(), "terminal close orphaned its child"
        report = json.loads((repo / "runs/.install/desktop-latest.json").read_text())
        assert report["exit_code"] == 130 and report["finished_at"] >= report["started_at"]
        assert "CHILD_READY" in Path(report["log"]).read_text()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if (repo / "child.pid").exists():
            try:
                os.killpg(int((repo / "child.pid").read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.stdout is not None:
            process.stdout.close()
