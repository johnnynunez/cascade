"""Captured evidence integrity and read-only replay protocol checks."""

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from cascade.agent.decision import DecisionError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark/diagnostics"))
try:
    import decision_replay as replay
finally:
    sys.path.pop(0)


def test_captured_fixture_integrity_projection_and_valid_tool_schemas():
    fixture = replay.load_fixture(replay.DEFAULT_FIXTURE)
    assert len(fixture["cases"]) == 10
    assert sum(c["expected_complete"] for c in fixture["cases"]) == 3
    tree = ast.parse((ROOT / "src/cascade/skills/runtime.py").read_text())
    specs = next(ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "TOOL_SPECS")
    by_name = {s["name"]: s for s in specs}
    assert replay.PREPARED_TOOLS <= by_name.keys()
    for case in fixture["cases"]:
        state = replay.state_for(case)
        assert not replay.OMITTED_RESULT_FIELDS.intersection(state["result"])
        assert "expected_choice" not in json.dumps(state)
        assert state["context_before_operation"] == case["record"]["context"]
        for c in case["candidates"]:
            schema = by_name[c["tool"]]["parameters"]
            assert set(schema["required"]) <= c["arguments"].keys() <= schema["properties"].keys()


@pytest.mark.parametrize("mutation,pattern", [
    (lambda c: c["record"]["result"].update(ok=False), "hash mismatch"),
    (lambda c: c["candidates"][0].update(tool="home"), "supported tool"),
    (lambda c: c["candidates"][0]["arguments"].update(expected_choice="leak"), "Prepared|Scoring"),
    (lambda c: c.update(expected_complete=1), "boolean"),
    (lambda c: c["candidates"][1].update(id=c["candidates"][0]["id"]), "unique"),
])
def test_invalid_fixture_fails_before_inference(tmp_path, mutation, pattern):
    fixture = json.loads(replay.DEFAULT_FIXTURE.read_text())
    mutation(fixture["cases"][0])
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture))
    with pytest.raises(ValueError, match=pattern):
        replay.load_fixture(path)


def test_projection_rejects_nested_scoring_metadata(tmp_path):
    fixture = json.loads(replay.DEFAULT_FIXTURE.read_text())
    case = fixture["cases"][0]
    case["record"]["result"]["history"] = [{"expected_complete": False}]
    case["source"]["record_sha256"] = replay.digest(case["record"])
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture))
    with pytest.raises(ValueError, match="Scoring"):
        replay.load_fixture(path)


class Stub:
    endpoint = "stub-no-inference"
    model = "stub"
    timeout_s = 1


def test_replay_records_error_and_selected_arguments_atomically(tmp_path):
    fixture = replay.load_fixture(replay.DEFAULT_FIXTURE)
    fixture["cases"] = fixture["cases"][:2]
    class Backend(Stub):
        def evaluate(self, case):
            if case["expected_complete"]:
                raise DecisionError("test failure")
            return {"choice": {"choice": case["expected_choice"]}, "completion": {"boolean": False}, "latency_ms": 3}
    path = tmp_path / "report.json"
    report = replay.run_replay(Backend(), fixture, path, "fixture-sha")
    assert json.loads(path.read_text()) == report
    assert report["run_complete"] and report["summary"]["request_errors"] == 1
    assert report["summary"]["correct_choices"] == 1
    assert report["summary"]["false_success_reports"] == 0
    assert report["cases"][0]["chosen_arguments"]["success"] is False
    assert "noul" not in report["cases"][0]["completion"]


def test_interruption_replaces_old_report_before_first_request(tmp_path):
    fixture = replay.load_fixture(replay.DEFAULT_FIXTURE)
    class Backend(Stub):
        def evaluate(self, case):
            raise KeyboardInterrupt
    path = tmp_path / "report.json"
    path.write_text('{"run_complete":true}')
    with pytest.raises(KeyboardInterrupt):
        replay.run_replay(Backend(), fixture, path, "fixture-sha")
    report = json.loads(path.read_text())
    assert report["run_complete"] is False and report["cases"] == []
    assert len(report["planned_case_ids"]) == 10


@pytest.mark.parametrize("arguments", [[], {"candidate": [], "goal_complete": False},
                                         {"candidate": "unknown", "goal_complete": False}])
def test_planner_rejects_malformed_selection_without_actuating(arguments):
    backend = object.__new__(replay.PlannerBackend)
    backend.model = "stub"
    backend.client = SimpleNamespace(chat=lambda **kw: SimpleNamespace(
        tool_calls=[SimpleNamespace(name="select_candidate", arguments=arguments)]))
    with pytest.raises(DecisionError, match="invalid"):
        backend.evaluate(replay.load_fixture(replay.DEFAULT_FIXTURE)["cases"][0])
