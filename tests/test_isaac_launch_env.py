"""Isaac launch environment/signal adapter; real children but no Kit/GPU."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def adapter():
    path = ROOT / "scripts/isaac_launch.py"
    assert path.exists(), "missing isolated source launcher"
    spec = importlib.util.spec_from_file_location("isaac_launch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_isaac_environment_does_not_inherit_agent_python_or_profiler(tmp_path):
    module = adapter()
    source = tmp_path / "source"
    env = module.clean_environment({
        "HOME": str(tmp_path), "USER": "operator", "DISPLAY": ":1", "XAUTHORITY": "/auth",
        "XDG_RUNTIME_DIR": "/runtime", "OMNI_KIT_ACCEPT_EULA": "YES", "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONEXE": "/agent/python", "PYTHONHOME": "/agent", "PYTHONPATH": "/agent/src",
        "VIRTUAL_ENV": "/agent", "CONDA_PREFIX": "/conda", "LD_LIBRARY_PATH": "/agent/lib",
        "LD_PRELOAD": "/profiler/injected.so", "NSYS_PROFILING_SESSION_ID": "agent-session",
        "CASCADE_PHYSICS_DEVICE": "cuda:0", "ISAACSIM_PATH": str(source),
        "SECRET_API_TOKEN": "should-not-leak",
    }, source=str(source))
    for key in ("PYTHONEXE", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX", "LD_LIBRARY_PATH", "NSYS_PROFILING_SESSION_ID", "SECRET_API_TOKEN"):
        assert key not in env
    assert env.get("LD_PRELOAD") != "/profiler/injected.so"
    assert env["DISPLAY"] == ":1" and env["XAUTHORITY"] == "/auth"
    assert env["ISAACSIM_PATH"] == str(source)
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["CASCADE_PHYSICS_DEVICE"] == "cuda:0"
    assert env["OMNI_KIT_ACCEPT_EULA"] == "YES"


def test_environment_adapter_does_not_create_eula_consent():
    module = adapter()
    env = module.clean_environment({"HOME": "/home/test"}, source=None)
    assert "OMNI_KIT_ACCEPT_EULA" not in env


def test_launch_adapter_runs_real_child_with_exit_status_and_sanitized_env(tmp_path):
    adapter()
    child = tmp_path / "child.py"
    child.write_text('import os,json,sys; print(json.dumps(dict(os.environ))); sys.exit(17)\n')
    result = subprocess.run([sys.executable, str(ROOT / "scripts/isaac_launch.py"), "--python", sys.executable, "--", str(child)],
                            env={**os.environ, "PYTHONEXE": "/agent/python", "VIRTUAL_ENV": "/agent",
                                 "CASCADE_REQUIRE_CUDA": "1", "CASCADE_ISAAC_DT": "0.008333333333333333",
                                 "PAAI_CAMERA_VIDEO_CONFIG": str(tmp_path / "optional-video.json")},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 17
    env = json.loads(result.stdout)
    assert "PYTHONEXE" not in env and "VIRTUAL_ENV" not in env
    assert env["CASCADE_REQUIRE_CUDA"] == "1"
    assert env["CASCADE_ISAAC_DT"] == "0.008333333333333333"
    assert "PAAI_CAMERA_VIDEO_CONFIG" not in env


def test_term_reaches_the_source_wrappers_child_process(tmp_path):
    adapter()
    marker = tmp_path / "child.json"
    worker = tmp_path / "worker.py"
    worker.write_text('import os,json,pathlib,time\n' + f'pathlib.Path({str(marker)!r}).write_text(json.dumps({{"pid":os.getpid()}}))\n' + 'time.sleep(60)\n')
    shell = tmp_path / "python.sh"
    # Like Isaac's python.sh: it spawns Python, it does NOT exec it.
    shell.write_text(f'#!/bin/bash\n"{sys.executable}" "$@"\n')
    shell.chmod(0o755)
    process = subprocess.Popen([sys.executable, str(ROOT / "scripts/isaac_launch.py"), "--python", str(shell), "--", str(worker)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pid = None
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.02)
        pid = json.loads(marker.read_text())["pid"]
        process.terminate()
        assert process.wait(timeout=10) != 0
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True).stdout.strip()
            if not status or "Z" in status:
                break
            time.sleep(0.02)
        assert not status or "Z" in status, "stopping source wrapper left its Kit child alive"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
