"""CPU proof budgets remain bounded and do not bypass host verification."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def proof_module():
    path = Path(__file__).resolve().parents[1] / "scripts/demo_proof.py"
    spec = importlib.util.spec_from_file_location("proof_timeout_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("scale,expected", [(None, 120), ("2", 240), ("4", 480)])
def test_cli_and_process_budgets_scale_together(monkeypatch, tmp_path, scale, expected):
    p = proof_module()
    monkeypatch.delenv("CASCADE_PROOF_TIMEOUT_SCALE", raising=False)
    if scale is not None:
        monkeypatch.setenv("CASCADE_PROOF_TIMEOUT_SCALE", scale)
    calls = []
    envelope = {"status": "ok", "result": {"meta": {"agentMeta": {
        "sessionId": "s", "provider": "local", "model": "vision"}}}}

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(envelope), stderr="")

    monkeypatch.setattr(p.subprocess, "run", run)
    p.agent_turn("s", "local/vision", "OK", tmp_path / "turn.json", 120)
    command, kwargs = calls[0]
    assert command[command.index("--timeout") + 1] == str(expected)
    assert kwargs["timeout"] == expected + 30
    # A longer budget never permits a different effective model.
    envelope["result"]["meta"]["agentMeta"]["model"] = "different"
    with pytest.raises(p.ProofError, match="model changed"):
        p.agent_turn("s", "local/vision", "OK", tmp_path / "other.json", 120)


@pytest.mark.parametrize("value", ["0", "5", "nan", "inf", "bad"])
def test_invalid_scale_refuses_before_sending_chat(monkeypatch, tmp_path, value):
    p = proof_module()
    monkeypatch.setenv("CASCADE_PROOF_TIMEOUT_SCALE", value)
    monkeypatch.setattr(p.subprocess, "run", lambda *a, **kw: pytest.fail("no chat allowed"))
    with pytest.raises(p.ProofError, match="finite and between"):
        p.agent_turn("s", "local/vision", "OK", tmp_path / "turn.json", 120)
