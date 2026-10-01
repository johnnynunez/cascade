"""Synthetic receipt integrity regressions; these fixtures claim no live evidence."""
import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "native_confirmed", Path(__file__).resolve().parents[1] / "benchmark/diagnostics/spark_native_confirmed.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def update(path, mutate):
    data = json.loads(path.read_text())
    mutate(data)
    write(path, data)


@pytest.fixture
def evidence(tmp_path):
    repo = tmp_path / "original"
    session = "cascade-proof-" + "a" * 32
    state = repo / "runs/.launch/profile-cascade-demo"
    directory = state / session
    run = repo / ("runs/mcp_12_" + "b" * 32)
    trace = run / "trace.jsonl"
    proof_path = state / "proof.json"
    model = "custom-local/Qwen/Test"
    proof = {"verified": True, "sim": "isaac", "profile": "cascade-demo", "session_id": session,
             "model": model, "started_at": 100, "trace": str(trace), "evidence_dir": str(directory),
             "process": {"repo": str(repo), "run_dir": str(run), "state_dir": str(state),
                         "pid": 12, "role": "mcp", "profile": "cascade-demo", "registered_at": 101,
                         "owner": "c" * 32, "instance_id": "d" * 32},
             "props_reset": sorted(gate.PROPS), "cases": []}
    rows = []

    def make_case(path, obj, destination, start):
        checks = {name: True for name in gate.COMMON_CHECKS}
        if obj.endswith("_cube"):
            checks["whole_cube_inside_green_square_throughout_settle"] = True
        else:
            checks.update({name: True for name in gate.CONVEX_CHECKS})
            checks["whole_convex_collider_inside_" + destination.replace(" ", "_") + "_and_supported"] = True
            if obj == "tomato_can":
                checks["tomato_can_settles_upright"] = True
        physics = {"pass": True, "errors": [], "object_name": obj, "destination_name": destination,
                   "expected_props": sorted(gate.PROPS), "checks": checks,
                   "pick": {"pass": True, "physics_pass": True, "camera_pass": True, "failed_checks": [],
                            "object": obj, "destination_name": destination, "checks": {"real_lift": True}},
                   "reset": {name: {"pass": True} for name in gate.PROPS},
                   "supported_destination_entry": {"pass": True},
                   "event_cameras": {"pass": True, "cameras": {
                       name: {"pass": True, "phase_advancement": {"pick": True, "reset": True}, "unique_jpeg_hashes": 4}
                       for name in ("cam0", "side", "proof")}}}
        phases = {name: {"unix": start + n * 10, "monotonic": start + n * 10 - 50}
                  for n, name in enumerate(gate.PHASES)}
        metadata = {"complete": True, "errors": [], "object_name": obj, "destination_name": destination,
                    "marks": phases, "samples": 50, "expected_props": sorted(gate.PROPS)}
        write(path / "physics/gpu-physical-audit.json", physics)
        write(path / "physics/phases.json", phases)
        write(path / "physics/metadata.json", metadata)
        case = {"object": obj, "destination": destination, "physics": physics,
                "props_reset": sorted(gate.PROPS), "evidence_dir": str(path)}
        for skill, offset in (("pick_and_place", 5), ("reset_scene", 25)):
            result = ({"ok": True, "verified": True, "picked": obj, "destination": destination,
                       "return_home": {"attempted": True, "ok": True, "at": "home"},
                       "postcondition": {"status": "confirmed", "channel": "physics",
                                         "skill": "pick_and_place", "kind": "relocated",
                                         "measured": {"object": obj, "destination": destination}}}
                      if skill == "pick_and_place" else
                      {"ok": True, "world": "isaac", "props_reset": sorted(gate.PROPS)})
            rows.append({"step": len(rows), "t": start + offset, "skill": skill,
                         "args": {"object": obj, "destination": destination} if skill == "pick_and_place" else {},
                         "result": result, "duration_ms": 1000})
        return case

    def native(path, skill, index):
        identifier = f"{index:08x}-1111-2222-3333-444444444444"
        terminal = {"runId": identifier, "turnId": identifier, "sessionId": session, "rerouted": False,
                    "requested": {"provider": "custom-local", "model": "Qwen/Test"},
                    "effective": {"provider": "custom-local", "model": "Qwen/Test"},
                    "successfulToolNames": ["cascade__" + skill]}
        write(path, {"runId": identifier, "status": "ok", "result": {"meta": {
            "aborted": False, "agentMeta": {"sessionId": session, "provider": "custom-local", "model": "Qwen/Test",
                                           "terminalReceipt": terminal},
            "toolSummary": {"calls": 1, "failures": 0, "tools": ["cascade__" + skill]}}}})

    for n, (obj, destination) in enumerate(gate.PROOF_CASES, 1):
        path = directory / f"case-{n}"
        proof["cases"].append(make_case(path, obj, destination, 110 + (n - 1) * 40))
        native(path / "02-pick.json", "pick_and_place", 2 * n)
        native(path / "03-reset.json", "reset_scene", 2 * n + 1)
    write(proof_path, proof)

    campaign_path = repo / "runs/campaign/campaign.json"
    campaign = {"schema_version": 1, "pass": True, "run_complete": True, "source_unchanged": True,
                "protected_receipts_unchanged": True, "errors": [], "engine": "physx", "repo": str(repo),
                "planned_cases": [list(case) for case in gate.CAMPAIGN_CASES],
                "source_commit": "e" * 40, "source_sha256": {"src/example.py": "f" * 64},
                "protected_receipts_sha256": {"runs/.launch/profile-cascade-demo/proof.json": gate.digest(proof_path.read_bytes())},
                "binding": {key: copy.deepcopy(proof[key]) for key in ("session_id", "model", "process", "trace")},
                "started_at": 200, "finished_at": 405, "cases": []}
    idle = {"enabled": True, "ready": True, "busy": False, "agent": "cascade-demo"}
    for n, (obj, destination) in enumerate(gate.CAMPAIGN_CASES, 1):
        start = 210 + (n - 1) * 40
        path = campaign_path.parent / f"case-{n}-{obj}"
        case = make_case(path, obj, destination, start)
        case.pop("evidence_dir")
        case.update({"pass": True, "reset_confirmed": True, "engine_matches_every_sample": True,
                     "pick_trace_confirmed": True, "errors": [], "observer_errors": [], "uncertain_order": False,
                     "started_at": start - 1, "finished_at": start + 31})
        for phase, skill, filename, offset in (("pick", "pick_and_place", "01-pick.json", 0),
                                             ("reset", "reset_scene", "02-reset.json", 20)):
            identifier = f"{n * 100 + offset:032x}"
            turn = {"status": "done", "required_tool": skill, "initial": idle,
                    "accepted": {"id": identifier, "status": "running"},
                    "last_response": {**idle, "order": {"id": identifier, "status": "done", "tool_failures": 0,
                                                         "tools": [skill], "finished_at": start + offset + 8}},
                    "started_at": start + offset + 1, "finished_at": start + offset + 9}
            case[phase] = turn
            write(path / filename, turn)
        write(path / "trace-evidence.json", rows[-2:])
        write(path / "receipt.json", case)
        campaign["cases"].append(case)
    write(campaign_path, campaign)
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {"repo": repo, "proof": proof_path, "campaign": campaign_path, "trace": trace,
            "directory": directory, "rows": rows}


def trace_rows(evidence, mutate):
    rows = [json.loads(line) for line in evidence["trace"].read_text().splitlines()]
    mutate(rows)
    evidence["trace"].write_text("".join(json.dumps(row) + "\n" for row in rows))


def errors(report):
    return " ".join(report["errors"] + [error for item in report["proof_native_cases"] + report["campaign_native_cases"]
                                          for error in item["errors"]])


def test_full_gate_requires_seven_native_results_and_never_mutates_receipts(evidence):
    before = {path: path.read_bytes() for path in evidence["repo"].rglob("*") if path.is_file()}
    # mtime is deliberately before the first event and cannot bound acceptance.
    os.utime(evidence["proof"], (1, 1))
    report = gate.validate(evidence["proof"], evidence["campaign"])
    assert report["pass"], errors(report)
    assert len(report["proof_native_cases"]) == 2
    assert len(report["campaign_native_cases"]) == 5
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("mutation,reason", [
    (lambda rows: rows[0]["result"].update(verified=False), "verified=true"),
    (lambda rows: rows[0]["result"]["postcondition"].update(status="unverified"), "confirmed physics"),
    (lambda rows: rows[0]["result"]["postcondition"].update(channel="vision"), "confirmed physics"),
    (lambda rows: rows[0]["result"]["postcondition"].update(kind="holding"), "another action"),
    (lambda rows: rows[0]["result"]["postcondition"]["measured"].update(object="lemon"), "another action"),
    (lambda rows: rows[0]["result"]["return_home"].update(ok=False), "return home did not complete"),
    (lambda rows: rows[0]["result"].pop("return_home"), "return_home must be an object"),
    (lambda rows: rows[0]["result"].update(ok=1), "ok=true"),
    (lambda rows: rows[0]["args"].update(destination="open box"), "arguments mismatch"),
    (lambda rows: rows[0]["result"].update(destination="open box"), "result identity"),
    (lambda rows: rows[0].update(t=109), "exactly one pick"),
    (lambda rows: rows[0].update(duration_ms=6000), "began before"),
    (lambda rows: rows[1].update(step=0), "duplicated or unordered"),
    (lambda rows: rows[1]["result"].update(world="mujoco"), "same-world reset"),
])
def test_full_external_pass_cannot_rescue_native_failure_or_unbound_event(evidence, mutation, reason):
    trace_rows(evidence, mutation)
    report = gate.validate(evidence["proof"])
    assert not report["pass"]
    assert reason in errors(report)


def test_an_extra_action_between_observer_phases_cannot_hide(evidence):
    def add(rows):
        rows.insert(1, {"step": 1, "t": 125, "skill": "move_home", "args": {}, "result": {"ok": True}, "duration_ms": 100})
        for step, row in enumerate(rows):
            row["step"] = step
    trace_rows(evidence, add)
    report = gate.validate(evidence["proof"])
    assert not report["pass"]
    assert "extra action" in errors(report)


@pytest.mark.parametrize("mutation,reason", [
    (lambda audit: audit["checks"].pop("two_jaw_contacts_during_real_lift"), "lacks required"),
    (lambda audit: audit["checks"].update(scene_content_sha256_matches=False), "failed/nonboolean"),
    (lambda audit: audit["event_cameras"]["cameras"]["side"].update(unique_jpeg_hashes=1), "did not advance"),
    (lambda audit: audit["reset"].pop("orange"), "every prop"),
    (lambda audit: audit.update(object_name="pink_cube"), "another case"),
])
def test_a_pass_bit_is_not_a_full_physical_audit(evidence, mutation, reason):
    update(evidence["directory"] / "case-1/physics/gpu-physical-audit.json", mutation)
    update(evidence["proof"], lambda proof: mutation(proof["cases"][0]["physics"]))
    report = gate.validate(evidence["proof"])
    assert not report["pass"]
    assert reason in errors(report)


def test_stale_or_other_session_native_receipt_fails(evidence):
    update(evidence["directory"] / "case-1/02-pick.json",
           lambda turn: turn["result"]["meta"]["agentMeta"].update(sessionId="another-session"))
    report = gate.validate(evidence["proof"])
    assert not report["pass"]
    assert "session/model mismatch" in errors(report)


def test_partial_real_shape_reports_completed_case_without_accepting_missing_case(evidence):
    def partial(proof):
        proof["verified"] = False
        proof["cases"].pop()
        proof.pop("props_reset")
    update(evidence["proof"], partial)
    report = gate.validate(evidence["proof"])
    assert not report["pass"]
    assert report["proof_native_cases"][0]["pass"]
    assert not report["proof_native_cases"][1]["pass"]
    assert "proof is incomplete" in errors(report)


def test_archive_relocation_preserves_original_bytes_and_rejects_escape(evidence, tmp_path):
    archive = tmp_path / "archive"
    original = evidence["repo"]
    original.rename(archive)
    proof = archive / evidence["proof"].relative_to(original)
    campaign = archive / evidence["campaign"].relative_to(original)
    report = gate.validate(proof, campaign, evidence_root=archive)
    assert report["pass"], errors(report)
    assert report["process"]["repo"] == str(original)
    assert str(proof) in report["input_sha256"]
    update(proof, lambda data: data.update(trace="/outside/trace.jsonl"))
    assert not gate.validate(proof, evidence_root=archive)["pass"]


@pytest.mark.parametrize("target,mutation,reason", [
    ("campaign", lambda data: data["binding"].update(session_id="wrong"), "another proof"),
    ("campaign", lambda data: data["protected_receipts_sha256"].update({"runs/.launch/profile-cascade-demo/proof.json": "0" * 64}), "protected proof hash"),
    ("campaign", lambda data: data["cases"][0].update(finished_at=500), "saved receipt"),
])
def test_campaign_cannot_borrow_another_proof_or_case(evidence, target, mutation, reason):
    update(evidence[target], mutation)
    report = gate.validate(evidence["proof"], evidence["campaign"])
    assert not report["pass"]
    assert reason in errors(report)


def test_campaign_order_and_trace_must_both_be_bound(evidence):
    case_path = evidence["campaign"].parent / "case-1-green_cube"
    mutation = lambda data: data["pick"]["last_response"]["order"].update(id="9" * 32)
    update(evidence["campaign"], lambda data: mutation(data["cases"][0]))
    update(case_path / "receipt.json", mutation)
    update(case_path / "01-pick.json", lambda data: data["last_response"]["order"].update(id="9" * 32))
    report = gate.validate(evidence["proof"], evidence["campaign"])
    assert not report["pass"]
    assert "order ID" in errors(report)


def test_saved_trace_cannot_replace_current_failed_native_result(evidence):
    trace_rows(evidence, lambda rows: rows[4]["result"].update(verified=False))
    report = gate.validate(evidence["proof"], evidence["campaign"])
    assert not report["pass"]
    assert "saved case trace differs" in errors(report)


@pytest.mark.parametrize("contents", ["{broken", '{"verified":true,"verified":false}', '{"started_at":NaN}',
                                       '{"started_at":1e999}', "[]", "[" * 2000 + "0" + "]" * 2000])
def test_malformed_inputs_write_failure_and_exit_one(tmp_path, contents):
    proof, output = tmp_path / "proof.json", tmp_path / "failure.json"
    proof.write_text(contents)
    assert gate.main(["--proof", str(proof), "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert not report["pass"] and report["errors"]
    assert proof.read_text() == contents


def test_output_never_overwrites_proof_or_earlier_report(evidence, tmp_path):
    proof = evidence["proof"]
    before = proof.read_bytes()
    assert gate.main(["--proof", str(proof), "--output", str(proof)]) == 1
    assert proof.read_bytes() == before
    output = tmp_path / "report.json"
    assert gate.main(["--proof", str(proof), "--output", str(output)]) == 0
    saved = output.read_bytes()
    assert gate.main(["--proof", str(proof), "--output", str(output)]) == 1
    assert output.read_bytes() == saved


def test_concurrent_input_change_fails_closed(evidence, monkeypatch):
    original = gate.Inputs.unchanged
    def changed(inputs):
        evidence["proof"].write_bytes(evidence["proof"].read_bytes() + b" ")
        original(inputs)
    monkeypatch.setattr(gate.Inputs, "unchanged", changed)
    report = gate.validate(evidence["proof"])
    assert not report["pass"]
    assert "evidence changed" in errors(report)
