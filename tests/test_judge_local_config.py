"""A local judge must not inherit gateway routing or certify missing scores."""
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

from cascade.eval.progress_judge import OpenAICompatJudge, judge_run


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/judge_run.py"


def recorded_run(tmp_path, *, frames=True, count=1):
    run = tmp_path / "run"
    run.mkdir()
    if frames:
        for name, value in (("before.jpg", 10), ("after.jpg", 200)):
            cv2.imwrite(str(run / name), np.full((24, 32, 3), value, np.uint8))
    rows = [{"step": i, "skill": "pick_and_place", "args": {"object": "cube"},
             "keyframe_before": "before.jpg", "keyframe_after": "after.jpg",
             "result": {"ok": True, "postcondition": {"status": "confirmed", "channel": "physics"}}}
            for i in range(count)]
    (run / "trace.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    return run


def test_local_config_replaces_gateway_defaults_and_cli_wins(tmp_path):
    run = recorded_run(tmp_path)
    config = tmp_path / "judge.json"
    config.write_text(json.dumps({"backend": "fake", "score": -0.5}))
    result = subprocess.run([sys.executable, str(SCRIPT), str(run), "--config", str(config),
                             "--fake-score", "0.25", "--strict"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    verdict = json.loads((run / "judge.json").read_text())
    assert verdict["judge"] == "fake"
    assert verdict["steps"][0]["hop"] == 0.25
    assert verdict["confusion"]["n_scored"] == 1


def test_strict_refuses_unscored_physics_confirmation(tmp_path):
    run = recorded_run(tmp_path, frames=False)
    result = subprocess.run([sys.executable, str(SCRIPT), str(run), "--judge", "fake", "--strict"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 4, result.stdout + result.stderr
    verdict = json.loads((run / "judge.json").read_text())
    assert verdict["confusion"]["n_scored"] == 0
    assert verdict["steps"][0]["error"] == "missing keyframe pair"


def test_raw_judge_output_is_retained_and_never_reused_after_transport_error(tmp_path, monkeypatch):
    run = recorded_run(tmp_path, count=2)
    create = Mock(side_effect=[SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="<score>+25%</score>"))]), RuntimeError("offline")])
    module = ModuleType("openai")
    module.OpenAI = Mock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    module.DefaultHttpxClient = Mock()
    monkeypatch.setitem(sys.modules, "openai", module)
    judge = OpenAICompatJudge(model="local-vision", base_url="http://127.0.0.1:9/v1", api_key="unused",
                             extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    module.DefaultHttpxClient.assert_called_once_with(trust_env=False, timeout=120.0)
    assert module.OpenAI.call_args.kwargs["http_client"] is module.DefaultHttpxClient.return_value
    verdict = judge_run(run, judge)
    assert verdict.steps[0].raw == "<score>+25%</score>"
    assert verdict.steps[1].raw is None
    assert verdict.steps[1].hop is None
    assert "offline" in verdict.steps[1].error
    assert create.call_args.kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert verdict.steps[1].response_metadata is None
