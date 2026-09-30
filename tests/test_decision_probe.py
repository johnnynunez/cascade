"""Fixture/scoring checks. Stub answers here are not model-quality evidence."""

import importlib.util
import json
from pathlib import Path

import pytest

from cascade.agent.decision import ChoiceAnswer, DecisionError, DecisionResponse, NoulAnswer


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("decision_probe", ROOT / "benchmark/diagnostics/decision_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_fixture_uses_real_tools_plus_explicit_abstention():
    from cascade.skills.runtime import TOOL_SPECS

    fixture = probe.load_fixture(probe.DEFAULT_FIXTURE)
    real_names = {s["name"] for s in TOOL_SPECS}
    assert set(fixture["criteria"]) - {"defer_to_planner"} <= real_names
    assert len(fixture["cases"]) == 32
    assert sum(c["language"] == "en" for c in fixture["cases"]) == 16
    assert sum(c["language"] == "es" for c in fixture["cases"]) == 16


def test_pilot_sends_only_state_and_questions_and_keeps_errors(tmp_path):
    fixture = probe.load_fixture(probe.DEFAULT_FIXTURE)
    chosen = fixture["cases"][:2]
    calls = []

    class StubClient:
        endpoint = "http://127.0.0.1:1/v1/systemone"
        model = "test-double-not-a-model"
        timeout_s = 1

        def evaluate(self, state, questions):
            calls.append((state, questions))
            assert set(state) == set(chosen[len(calls) - 1]["state"])
            assert "expected_choice" not in json.dumps(state)
            assert "expected_complete" not in json.dumps(state)
            if len(calls) == 2:
                raise DecisionError("test transport failure; no retry")
            return DecisionResponse(self.model, "stub", {
                "next_skill": ChoiceAnswer("get_observation", {"get_observation": 1.0}, 1.0),
                "goal_complete": NoulAnswer(0.2),
            }, 12.0, "request-hash", "response-hash", None)

    output = tmp_path / "report.json"
    report = probe.run_probe(StubClient(), fixture, output=output, fixture_sha256="fixture-hash", cases=chosen)
    assert len(calls) == 2
    assert json.loads(output.read_text()) == report
    assert report["planned_case_ids"] == [c["id"] for c in chosen]
    assert report["run_complete"] is True
    assert report["finished_at"] >= report["started_at"]
    assert report["summary"]["cases"] == 2
    assert report["summary"]["valid_responses"] == 1
    assert report["summary"]["request_errors"] == 1
    assert report["summary"]["choice_accuracy_all_cases"] == 0.5
    assert report["summary"]["completion_brier_valid_cases"] == pytest.approx(0.04)
    assert report["cases"][1]["status"] == "error"


def test_scoring_does_not_accept_partial_or_empty_choices():
    rows = [
        {"language": "en", "category": "routing", "status": "ok", "choice_correct": False,
         "completion_correct": False, "completion_squared_error": 0.81,
         "latency_ms": 10, "reported_model": "stub"},
        {"language": "es", "category": "routing", "status": "error"},
    ]
    summary = probe.summarize(rows)
    assert summary["correct_choices"] == 0
    assert summary["choice_accuracy_all_cases"] == 0
    assert summary["groups"]["language"]["es"]["valid_responses"] == 0
    assert summary["median_latency_ms_valid"] == 10
    assert summary["p95_latency_ms_valid"] == 10


def test_official_cli_requires_key_instead_of_substituting_a_local_model(monkeypatch, tmp_path):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(SystemExit) as caught:
        probe.main(["--base-url", "https://api.typesafe.ai", "--model", "jev-latest",
                    "--output", str(tmp_path / "no-result.json")])
    assert caught.value.code == 2
    assert not (tmp_path / "no-result.json").exists()


@pytest.mark.parametrize("nested", [
    {"observation": {"expected_choice": "get_observation"}},
    {"history": [{"expected_complete": False}]},
])
def test_fixture_rejects_nested_scoring_labels_before_inference(tmp_path, nested):
    fixture = json.loads(probe.DEFAULT_FIXTURE.read_text())
    fixture["cases"][-1]["state"].update(nested)
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture))
    with pytest.raises(ValueError, match="Expected labels"):
        probe.load_fixture(path)


def test_fixture_rejects_nonfinite_state_before_inference(tmp_path):
    fixture = json.loads(probe.DEFAULT_FIXTURE.read_text())
    fixture["cases"][-1]["state"]["depth_m"] = float("nan")
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture))
    with pytest.raises(ValueError):
        probe.load_fixture(path)


@pytest.mark.parametrize("interrupt_on", [1, 2])
def test_interrupted_pilot_keeps_planned_cases_and_never_looks_complete(tmp_path, interrupt_on):
    fixture = probe.load_fixture(probe.DEFAULT_FIXTURE)
    chosen = fixture["cases"][:2]
    output = tmp_path / "report.json"
    output.write_text('{"stale_report": true}')

    class InterruptedClient:
        endpoint = "http://127.0.0.1:1/v1/systemone"
        model = "test-double-not-a-model"
        timeout_s = 1
        calls = 0

        def evaluate(self, state, questions):
            self.calls += 1
            checkpoint = json.loads(output.read_text())
            assert checkpoint["planned_case_ids"] == [c["id"] for c in chosen]
            assert checkpoint["run_complete"] is False
            if self.calls == interrupt_on:
                raise KeyboardInterrupt
            raise DecisionError("test transport failure; no retry")

    with pytest.raises(KeyboardInterrupt):
        probe.run_probe(InterruptedClient(), fixture, output=output,
                        fixture_sha256="fixture-hash", cases=chosen)
    partial = json.loads(output.read_text())
    assert partial["run_complete"] is False
    assert "finished_at" not in partial
    assert partial["summary"]["cases"] == interrupt_on - 1
    assert partial["summary"]["request_errors"] == interrupt_on - 1


def test_empty_pilot_is_rejected_without_replacing_previous_report(tmp_path):
    fixture = probe.load_fixture(probe.DEFAULT_FIXTURE)
    output = tmp_path / "report.json"
    output.write_text('{"previous": true}')
    with pytest.raises(ValueError, match="at least one case"):
        probe.run_probe(None, fixture, output=output, fixture_sha256="unused", cases=[])
    assert json.loads(output.read_text()) == {"previous": True}
