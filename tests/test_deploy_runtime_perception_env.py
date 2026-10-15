"""deploy/runtime/runtime.py: the Brev kitchen's robot server asks its own brain.

The deployment serves its Qwen3.8 brain with llama.cpp on 127.0.0.1:8041
(`deploy/runtime/brain.py`, alias = the production model id) and starts the
robot MCP server with every CASCADE_* variable of `runtime_environment()`.
configs/demo.yaml identifies objects with the brain at
`${CASCADE_PERCEPTION_URL:-http://127.0.0.1:8080/v1}` -- a port this
deployment does not serve -- so the environment must name the deployment's
own brain and its served model. Stdlib only: deploy/runtime/tests needs
aiohttp, which the CI test job does not install.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "deploy" / "runtime"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"cascade_deploy_{name}_under_test", RUNTIME / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    """runtime.py with fixture paths (nothing under /data or /opt) and no subprocess."""
    for key in list(os.environ):
        if key.startswith(("PAAI_", "CASCADE_")):
            monkeypatch.delenv(key)
    module = _load("runtime")   # reads PAAI_MODEL_ID at import
    node = tmp_path / "bin" / "node"
    node.parent.mkdir(parents=True)
    node.write_text("fixture; never executed\n")
    monkeypatch.setenv("PAAI_NODE", str(node))
    monkeypatch.setenv("PAAI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PAAI_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("PAAI_INSTANCE", "fixture")
    monkeypatch.setattr(module, "private_run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "v24.16.0\n", ""))
    return module


def test_the_robot_server_identifies_objects_with_the_deployments_own_brain(runtime, monkeypatch):
    env = runtime.runtime_environment()
    assert env["CASCADE_PERCEPTION_URL"] == "http://127.0.0.1:8041/v1" == env["CASCADE_QWEN_BASE_URL"]
    assert env["CASCADE_PERCEPTION_MODEL"] == "Qwen/Qwen3.8-27B" == env["CASCADE_GPU_MODEL_ID"]
    # ... which is exactly the endpoint and alias the deployment's brain serves
    # (setup() builds the brain plan under this environment).
    for name, value in env.items():
        if name.startswith("CASCADE_"):
            monkeypatch.setenv(name, value)
    plan = _load("brain").make_plan(require_files=False)
    assert (plan["base_url"], plan["model_id"]) == (env["CASCADE_PERCEPTION_URL"], env["CASCADE_PERCEPTION_MODEL"])
