"""Launcher probe lifecycle and error visibility; no GPU success is inferred."""
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test_launch_delivery import launcher_boundary as _launcher_boundary
from test_launch_delivery import model_http_boundary as _model_http_boundary

launcher_boundary = _launcher_boundary
model_http_boundary = _model_http_boundary
REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("inspection_fails", [False, True])
def test_real_runtime_probe_shuts_down_every_component(monkeypatch, inspection_fails):
    from cascade.apps import demo

    events = []
    arm = object()

    def backends():
        events.append("inspect")
        if inspection_fails:
            raise RuntimeError("backend inspection failed")
        return {"test_double": True}

    runtime = SimpleNamespace(backends=backends, camera=SimpleNamespace(close=lambda: events.append("camera-only")))
    monkeypatch.setattr(demo, "build_runtime", lambda *a, **kw: (runtime, arm))

    def shutdown(actual_runtime, actual_arm):
        assert actual_runtime is runtime and actual_arm is arm
        events.append("shutdown")

    monkeypatch.setattr(demo, "shutdown_runtime", shutdown)
    monkeypatch.setenv("CASCADE_CAMERAS", "mock")
    monkeypatch.setenv("CASCADE_ARM", "mock")
    blocks = re.findall(r"<<'PYEOF'[^\n]*\n(.*?)\nPYEOF", (REPO / "scripts/launch.sh").read_text(), re.S)
    probes = [block for block in blocks if "rt.backends()" in block]
    assert len(probes) == 1
    code = compile(probes[0], "launcher-runtime-probe", "exec")
    if inspection_fails:
        with pytest.raises(RuntimeError, match="backend inspection failed"):
            exec(code, {})
    else:
        exec(code, {})
    assert events == ["inspect", "shutdown"]


def test_launcher_prints_captured_diagnostics_when_runtime_exits(launcher_boundary):
    h = launcher_boundary
    wrapper = Path(h["env"]["PY"])
    original = wrapper.with_name("python-original")
    wrapper.rename(original)
    wrapper.write_text(f"#!{sys.executable}\n" + f"original={str(original)!r}\n" + '''import os,sys,subprocess
args=sys.argv[1:]
if args and args[0]=='-':
    source=sys.stdin.read()
    if 'build_runtime' in source:
        print('RUNTIME_FAILURE_SENTINEL: native shutdown failed',flush=True)
        sys.exit(17)
    sys.exit(subprocess.run([original,*args],input=source,text=True).returncode)
os.execv(original,[original,*args])
''')
    wrapper.chmod(0o755)
    result = subprocess.run(h["command"], env=h["env"], text=True, capture_output=True, timeout=60)
    assert result.returncode != 0
    assert "RUNTIME_FAILURE_SENTINEL" in result.stdout + result.stderr
    assert "[launch] READY" not in result.stdout
