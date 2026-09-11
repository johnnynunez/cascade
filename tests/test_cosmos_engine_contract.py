"""Fail closed when a reasoner launch is requested as vLLM Omni."""
from __future__ import annotations

import pytest

from test_cosmos_serving_contract import load_helper, run_script


def test_reasoner_plan_names_actual_engine_and_component(tmp_path):
    helper = load_helper()
    plan = helper.serving_plan({"VENV": str(tmp_path / "cosmos")})
    assert plan["engine"] == "vllm"
    assert plan["model_component"] == "hf-reasoner-only"
    assert plan["vllm_omni"] is False
    assert "--omni" not in plan["serve_command"]


@pytest.mark.parametrize("engine", ["vllm-omni", "omni"])
def test_explicit_omni_request_is_not_silently_plain_vllm(tmp_path, engine):
    result = run_script(tmp_path, "--dry-run", overrides={"COSMOS_ENGINE": engine})
    assert result.returncode == 1, result.stdout + result.stderr
    assert "reasoner" in result.stderr and "vLLM-Omni" in result.stderr
    assert "UNEXPECTED_INSTALL" not in result.stderr
    assert not (tmp_path / "isolated cosmos").exists()


def test_reasoner_setup_cannot_reconcile_an_omni_environment(tmp_path, monkeypatch):
    helper = load_helper()
    venv = tmp_path / "already-omni"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").touch()
    (venv / "pyvenv.cfg").write_text("version = 3.12.13\n")
    metadata = venv / "lib/python3.12/site-packages/vllm_omni-0.29.0rc1.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: vllm-omni\nVersion: 0.29.0rc1\n")
    installs = []
    monkeypatch.setattr(helper.subprocess, "run", lambda *args, **kwargs: installs.append(args))
    with pytest.raises(RuntimeError, match="vLLM-Omni.*separate"):
        helper.ensure_environment(helper.serving_plan({"VENV": str(venv)}))
    assert installs == [], "do not replace the Omni environment's Transformers pin"
