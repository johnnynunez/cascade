#!/usr/bin/env python3
"""Read-only strict native outcome gate for a completed Spark proof/campaign.

This checks existing receipts, not the robot, and does not replay the physical
auditor. The full saved audits must pass in addition to each native placement.
Trace rows have no OpenClaw run/order ID: attribution uses the owner-bound trace,
unique step, exact case arguments and observer/turn time intervals. File mtimes
are never evidence. --evidence-root relocates the recorded repository's paths
to an archive with the same relative layout, without rewriting any receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys


PROOF_CASES = (("green_cube", "green square"), ("orange", "open box"))
CAMPAIGN_CASES = PROOF_CASES + (("pink_cube", "green square"),
                               ("lemon", "open box"), ("tomato_can", "green square"))
PROPS = {obj for obj, _ in CAMPAIGN_CASES}
READ_TOOLS = {"describe_scene", "localize_object", "camera_snapshot", "world_state"}
PHASES = ("pick_begin", "pick_end", "reset_begin", "reset_end")
COMMON_CHECKS = frozenset("""
complete_observation all_phase_boundaries live_prop_names_match_expected
spawn_prop_names_match_expected spawn_positions_finite_and_unchanged
all_props_physically_reset_and_settled scene_config_sha256_matches
prop_dimensions_match cube_collider_geometry_matches green_square_geometry_matches
scene_name_matches scene_assets_sha256_matches scene_content_sha256_matches
open_box_geometry_matches ordered_phases strict_pick_place_and_cameras
all_physics_tensors_cuda single_supported_engine gpu_backend_no_fallback
contact_sensor_and_actual_jaws_bound two_jaw_contacts_during_real_lift
reset_settle_window whole_footprint_enters_destination_with_bilateral_support
all_event_cameras_advance
""".split())
CONVEX_CHECKS = frozenset("""
convex_body_identity authored_convex_vertices_match convex_topology_matches
convex_body_and_collider_enabled convex_body_and_child_transforms_match
convex_tensor_mass_and_com_match convex_tensor_inertia_valid
physx_support_live_identity physx_support_source_matches physx_support_solid_subset
physx_support_geometry_unchanged
""".split())


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def mapping(value, label):
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def number(value, label):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    require(valid, f"{label} must be a finite nonnegative number")
    return value


def normalize(value):
    require(isinstance(value, str) and bool(value.strip()), "missing object/destination name")
    return " ".join(value.replace("_", " ").casefold().split())


def props(value, label):
    require(isinstance(value, list) and all(isinstance(x, str) for x in value)
            and len(value) == len(PROPS) and set(value) == PROPS,
            f"{label} must contain exactly the five kitchen props")


def checks_pass(value, required, label):
    checks = mapping(value, label)
    require(bool(checks) and required <= checks.keys(), f"{label} lacks required full-audit checks")
    require(all(flag is True for flag in checks.values()), f"{label} contains a failed/nonboolean check")


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(data):
    def invalid_constant(value):
        raise EvidenceError(f"nonfinite JSON value: {value}")
    def finite_float(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite JSON number")
        return result
    try:
        return json.loads(data, object_pairs_hook=unique_keys, parse_constant=invalid_constant, parse_float=finite_float)
    except RecursionError as error:
        raise EvidenceError("JSON nesting exceeds the parser limit") from error


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Inputs:
    """Read each file once; report exact hashes and detect concurrent mutation."""
    def __init__(self, evidence_root=None):
        self.data = {}
        self.original_root = None
        self.evidence_root = Path(evidence_root).resolve(strict=True) if evidence_root is not None else None

    def read(self, path):
        path = Path(path)
        if self.evidence_root is not None and self.original_root is not None and path.is_relative_to(self.original_root):
            path = self.evidence_root / path.relative_to(self.original_root)
        require(not path.is_symlink(), f"symlink evidence is not accepted: {path}")
        path = path.resolve(strict=True)
        if self.evidence_root is not None:
            require(path.is_relative_to(self.evidence_root), f"evidence escapes --evidence-root: {path}")
        if path not in self.data:
            require(path.is_file() and path.stat().st_size <= 128 * 1024 * 1024,
                    f"evidence must be a regular file of at most 128 MiB: {path}")
            with path.open("rb") as stream:
                data = stream.read(128 * 1024 * 1024 + 1)
            require(len(data) <= 128 * 1024 * 1024, f"evidence grew beyond size limit: {path}")
            self.data[path] = data
        return self.data[path]

    def json(self, path):
        return parse_json(self.read(path))

    def hashes(self):
        return {str(path): digest(data) for path, data in self.data.items()}

    def unchanged(self):
        for path, data in self.data.items():
            with path.open("rb") as stream:
                require(stream.read(len(data) + 1) == data, f"evidence changed during validation: {path}")


def recorded_path(value, label):
    require(isinstance(value, str) and Path(value).is_absolute() and ".." not in Path(value).parts,
            f"{label} must be an absolute path without traversal")
    return Path(value)


def read_trace(inputs, path):
    rows = [mapping(parse_json(line), "trace row")
            for line in inputs.read(path).splitlines() if line.strip()]
    require(bool(rows), "empty owner-bound trace")
    previous_step, previous_time = -1, -1
    for row in rows:
        step, timestamp = row.get("step"), number(row.get("t"), "trace t")
        require(type(step) is int and step > previous_step and timestamp >= previous_time,
                "trace step/time is duplicated or unordered")
        require(isinstance(row.get("skill"), str), "trace skill missing")
        mapping(row.get("args"), "trace args")
        mapping(row.get("result"), "trace result")
        previous_step, previous_time = step, timestamp
    return rows


def full_audit(inputs, case, directory, obj, destination):
    audit = mapping(case.get("physics"), "case physics")
    require(audit == inputs.json(directory / "physics/gpu-physical-audit.json"),
            "embedded physical audit differs from saved audit")
    require(audit.get("pass") is True and audit.get("errors") == [], "full physical audit did not pass")
    require(audit.get("object_name") == obj and audit.get("destination_name") == destination,
            "physical audit belongs to another case")
    props(audit.get("expected_props"), "physical expected_props")
    required = set(COMMON_CHECKS)
    if obj.endswith("_cube"):
        required.add("whole_cube_inside_green_square_throughout_settle")
    else:
        required.update(CONVEX_CHECKS)
        required.add("whole_convex_collider_inside_" + destination.replace(" ", "_") + "_and_supported")
        if obj == "tomato_can":
            required.add("tomato_can_settles_upright")
    checks_pass(audit.get("checks"), required, "full physical checks")
    pick = mapping(audit.get("pick"), "physical pick")
    require(all(pick.get(key) is True for key in ("pass", "physics_pass", "camera_pass"))
            and pick.get("failed_checks") == [], "physical pick/camera audit failed")
    require(pick.get("object") == obj and pick.get("destination_name") == destination,
            "physical pick identity mismatch")
    checks_pass(pick.get("checks"), set(), "physical pick checks")
    reset = mapping(audit.get("reset"), "physical reset")
    require(set(reset) == PROPS and all(mapping(x, "reset prop").get("pass") is True
                                      for x in reset.values()), "physical reset did not pass for every prop")
    require(mapping(audit.get("supported_destination_entry"), "supported entry").get("pass") is True,
            "physical transport/entry audit failed")
    cameras = mapping(audit.get("event_cameras"), "event cameras")
    channels = mapping(cameras.get("cameras"), "event camera channels")
    require(cameras.get("pass") is True and set(channels) == {"cam0", "side", "proof"},
            "full event camera audit missing/failed")
    for camera in channels.values():
        require(mapping(camera, "event camera").get("pass") is True
                and camera.get("phase_advancement") == {"pick": True, "reset": True}
                and type(camera.get("unique_jpeg_hashes")) is int and camera["unique_jpeg_hashes"] >= 2,
                "event cameras did not advance through pick/reset")
    phases = mapping(inputs.json(directory / "physics/phases.json"), "phases")
    metadata = mapping(inputs.json(directory / "physics/metadata.json"), "observer metadata")
    require(metadata.get("complete") is True and metadata.get("errors") == []
            and metadata.get("object_name") == obj and metadata.get("destination_name") == destination
            and metadata.get("marks") == phases, "observer metadata/phase/case mismatch")
    props(metadata.get("expected_props"), "observer expected_props")
    require(type(metadata.get("samples")) is int and metadata["samples"] >= 4,
            "observer has no complete sample record")
    require(set(phases) == set(PHASES), "incomplete phase boundaries")
    times = [number(mapping(phases[name], name).get("unix"), name + ".unix") for name in PHASES]
    monotonic = [number(phases[name].get("monotonic"), name + ".monotonic") for name in PHASES]
    require(all(a < b for a, b in zip(times, times[1:]))
            and all(a < b for a, b in zip(monotonic, monotonic[1:])), "unordered observer phases")
    return times


def tool_names(value, required, *, prefix=""):
    require(isinstance(value, list) and all(isinstance(x, str) for x in value)
            and prefix + required in value
            and set(value) <= {prefix + name for name in READ_TOOLS | {required}},
            "native turn lacks its required tool or includes another action")


def native_turn(inputs, path, proof, required, ids):
    envelope = mapping(inputs.json(path), "native envelope")
    require(envelope.get("status") == "ok", "native envelope did not complete")
    meta = mapping(mapping(envelope.get("result"), "native result").get("meta"), "native metadata")
    require(meta.get("aborted") is False, "native turn aborted/missing completion state")
    agent = mapping(meta.get("agentMeta"), "native agent metadata")
    require(agent.get("sessionId") == proof["session_id"]
            and f"{agent.get('provider')}/{agent.get('model')}" == proof["model"],
            "native turn session/model mismatch")
    receipt = mapping(agent.get("terminalReceipt"), "native terminal receipt")
    run_id = envelope.get("runId")
    require(isinstance(run_id, str) and re.fullmatch(r"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}", run_id)
            and run_id not in ids, "native runId missing/duplicated")
    ids.add(run_id)
    require(receipt.get("runId") == receipt.get("turnId") == run_id
            and receipt.get("sessionId") == proof["session_id"] and receipt.get("rerouted") is False,
            "terminal receipt run/session mismatch or rerouted")
    for key in ("requested", "effective"):
        model = mapping(receipt.get(key), "terminal " + key)
        require(f"{model.get('provider')}/{model.get('model')}" == proof["model"], "terminal model changed")
    summary = mapping(meta.get("toolSummary"), "native toolSummary")
    require(type(summary.get("calls")) is int and summary["calls"] >= 1
            and type(summary.get("failures")) is int and summary["failures"] == 0,
            "native toolSummary has failed/missing calls")
    tool_names(summary.get("tools"), required, prefix="cascade__")
    tool_names(receipt.get("successfulToolNames"), required, prefix="cascade__")
    return run_id


def select_action(rows, start, end, skill):
    current = [row for row in rows if start <= row["t"] <= end and row["skill"] not in READ_TOOLS]
    require(len(current) == 1 and current[0]["skill"] == skill,
            f"expected exactly one {skill} and no other action in its bounded interval")
    row = current[0]
    duration = number(row.get("duration_ms"), "action duration_ms") / 1000
    # Trace is written on completion; rounding to 0.1 ms permits 0.05 ms error.
    require(row["t"] - duration >= start - .0001, "action began before its bounded interval")
    return row


def native_outcome(rows, times, obj, destination, *, case_interval=None):
    pick = select_action(rows, times[0], times[1], "pick_and_place")
    require(normalize(pick["args"].get("object")) == normalize(obj)
            and normalize(pick["args"].get("destination")) == destination, "trace pick arguments mismatch")
    result = pick["result"]
    postcondition = mapping(result.get("postcondition"), "native postcondition")
    require(result.get("ok") is True and result.get("verified") is True
            and postcondition.get("status") == "confirmed" and postcondition.get("channel") == "physics",
            "native placement requires ok=true, verified=true, confirmed physics")
    require(normalize(result.get("picked")) == normalize(obj)
            and normalize(result.get("destination")) == destination, "native result identity mismatch")
    reset = select_action(rows, times[2], times[3], "reset_scene")
    require(reset["result"].get("ok") is True and reset["result"].get("world") == "isaac",
            "native same-world reset did not pass")
    props(reset["result"].get("props_reset"), "native reset props")
    start, end = case_interval or (times[0], times[-1])
    actions = [row["step"] for row in rows if start <= row["t"] <= end and row["skill"] not in READ_TOOLS]
    require(actions == [pick["step"], reset["step"]], "extra action outside the selected pick/reset intervals")
    return {"pick_step": pick["step"], "pick_t": pick["t"], "reset_step": reset["step"],
            "reset_t": reset["t"], "postcondition": postcondition,
            "pick_row_sha256": digest(json.dumps(pick, sort_keys=True, allow_nan=False).encode()),
            "reset_row_sha256": digest(json.dumps(reset, sort_keys=True, allow_nan=False).encode())}


def visitor_turn(inputs, path, embedded, required, ids):
    turn = mapping(embedded, "visitor turn")
    require(turn == inputs.json(path), "embedded visitor turn differs from its saved receipt")
    require(turn.get("status") == "done" and turn.get("required_tool") == required,
            "visitor turn incomplete/wrong required tool")
    initial = mapping(turn.get("initial"), "visitor initial")
    last = mapping(turn.get("last_response"), "visitor final")
    for status in (initial, last):
        require(status.get("enabled") is True and status.get("ready") is True
                and status.get("busy") is False and status.get("agent") == "cascade-demo",
                "visitor readiness/agent mismatch")
    accepted = mapping(turn.get("accepted"), "visitor acceptance")
    order = mapping(last.get("order"), "visitor order")
    identifier = accepted.get("id")
    require(isinstance(identifier, str) and re.fullmatch(r"[a-f0-9]{32}", identifier)
            and identifier not in ids and order.get("id") == identifier,
            "visitor order ID missing/reused/mismatched")
    ids.add(identifier)
    require(accepted.get("status") == "running" and order.get("status") == "done"
            and type(order.get("tool_failures")) is int and order["tool_failures"] == 0,
            "visitor order did not complete without tool failures")
    tool_names(order.get("tools"), required)
    start = number(turn.get("started_at"), "visitor started_at")
    finished = number(turn.get("finished_at"), "visitor finished_at")
    order_finished = number(order.get("finished_at"), "order finished_at")
    require(start < order_finished <= finished, "visitor order time mismatch")
    return identifier, start, order_finished


def case_identity(case, expected):
    mapping(case, "case")
    obj, destination = expected
    require(normalize(case.get("object")) == normalize(obj)
            and case.get("destination") == destination, "case list identity/order mismatch")
    props(case.get("props_reset"), "case props_reset")


def validate(proof_path, campaign_path=None, *, evidence_root=None):
    inputs = Inputs()
    report = {"schema_version": 1, "pass": False,
              "scope": "Additional native confirmed-physics gate over unchanged saved full audits; no actuation or physics replay",
              "trace_binding": "Owner-bound trace + unique step + exact arguments + observer/turn intervals; trace has no native run/order ID",
              "proof": str(proof_path), "errors": [], "proof_native_cases": [], "campaign_native_cases": []}
    ids = set()

    def case_result(collection, obj, destination, operation):
        item = {"object": obj, "destination": destination, "pass": False, "errors": []}
        collection.append(item)
        try:
            item.update(operation())
            item["pass"] = True
        except (OSError, ValueError, TypeError, KeyError) as error:
            item["errors"].append(f"{type(error).__name__}: {error}")

    try:
        report["checker_sha256"] = digest(Path(__file__).read_bytes())
        if evidence_root is not None:
            inputs.evidence_root = Path(evidence_root).resolve(strict=True)
            report["evidence_root"] = str(inputs.evidence_root)
        proof = mapping(inputs.json(proof_path), "proof")
        require(proof.get("sim") == "isaac" and proof.get("profile") == "cascade-demo", "native Isaac proof required")
        if proof.get("verified") is not True:
            report["errors"].append("proof is incomplete or verified is not true")
        session = proof.get("session_id")
        require(isinstance(session, str) and re.fullmatch(r"cascade-proof-[a-f0-9]{32}", session),
                "invalid proof session_id")
        require(isinstance(proof.get("model"), str) and "/" in proof["model"], "missing proof model")
        started = number(proof.get("started_at"), "proof started_at")
        process = mapping(proof.get("process"), "proof process")
        repo = recorded_path(process.get("repo"), "process repo")
        inputs.original_root = repo
        run_dir = recorded_path(process.get("run_dir"), "process run_dir")
        trace = recorded_path(proof.get("trace"), "proof trace")
        require(type(process.get("pid")) is int and process["pid"] > 0
                and process.get("role") == "mcp" and process.get("profile") == "cascade-demo"
                and run_dir.parent == repo / "runs" and run_dir.name.startswith(f"mcp_{process['pid']}_")
                and trace == run_dir / "trace.jsonl", "trace is not bound to the proof MCP process")
        for key in ("instance_id", "owner"):
            require(isinstance(process.get(key), str) and re.fullmatch(r"[a-f0-9]{32}", process[key]),
                    "invalid process " + key)
        require(number(process.get("registered_at"), "process registered_at") >= started,
                "MCP process predates this proof")
        evidence = recorded_path(proof.get("evidence_dir"), "proof evidence_dir")
        require(evidence.name == session and evidence.parent == recorded_path(process.get("state_dir"), "process state_dir"),
                "proof evidence directory/session mismatch")
        rows = read_trace(inputs, trace)
        report.update(proof_session=session, trace=str(trace), process=process)
        cases = proof.get("cases")
        require(isinstance(cases, list) and len(cases) <= len(PROOF_CASES), "proof has an invalid case list")
        if proof.get("verified") is True:
            props(proof.get("props_reset"), "proof props_reset")
        previous_end = started
        for number_, (obj, destination) in enumerate(PROOF_CASES, 1):
            case = cases[number_ - 1] if number_ <= len(cases) else None
            def check_proof_case():
                nonlocal previous_end
                require(case is not None, "completed case receipt is missing from proof")
                case_identity(case, (obj, destination))
                directory = evidence / f"case-{number_}"
                require(recorded_path(case.get("evidence_dir"), "case evidence_dir") == directory,
                        "case evidence directory mismatch")
                times = full_audit(inputs, case, directory, obj, destination)
                require(times[0] > previous_end, "proof case intervals overlap/predate proof")
                previous_end = times[-1]
                run_ids = {required: native_turn(inputs, directory / filename, proof, required, ids)
                           for required, filename in (("pick_and_place", "02-pick.json"), ("reset_scene", "03-reset.json"))}
                return {**native_outcome(rows, times, obj, destination), "run_ids": run_ids,
                        "phase_unix": dict(zip(PHASES, times)), "full_physical_audit_pass": True}
            case_result(report["proof_native_cases"], obj, destination, check_proof_case)

        if campaign_path is not None:
            campaign_path = Path(campaign_path)
            report["campaign"] = str(campaign_path)
            campaign = mapping(inputs.json(campaign_path), "campaign")
            require(all(campaign.get(key) is True for key in
                        ("pass", "run_complete", "source_unchanged", "protected_receipts_unchanged"))
                    and campaign.get("errors") == [], "campaign incomplete/failed/changed")
            require(campaign.get("engine") == "physx" and campaign.get("schema_version") == 1,
                    "unsupported campaign schema/engine")
            require(campaign.get("repo") == str(repo)
                    and campaign.get("planned_cases") == [list(case) for case in CAMPAIGN_CASES],
                    "campaign repo/planned cases mismatch")
            require(campaign.get("binding") == {key: proof[key] for key in ("session_id", "model", "process", "trace")},
                    "campaign is bound to another proof/process/trace")
            require(isinstance(campaign.get("source_commit"), str)
                    and re.fullmatch(r"[a-f0-9]{40}", campaign["source_commit"]), "missing campaign source commit")
            source_hashes = mapping(campaign.get("source_sha256"), "campaign source hashes")
            require(bool(source_hashes) and all(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value)
                                               for value in source_hashes.values()), "invalid campaign source hashes")
            protected = mapping(campaign.get("protected_receipts_sha256"), "protected receipt hashes")
            require(protected.get("runs/.launch/profile-cascade-demo/proof.json") == digest(inputs.read(proof_path)),
                    "campaign protected proof hash differs from supplied proof")
            report["campaign_source_commit"] = campaign["source_commit"]
            start = number(campaign.get("started_at"), "campaign started_at")
            end = number(campaign.get("finished_at"), "campaign finished_at")
            require(previous_end < start < end, "campaign does not follow proof")
            campaign_cases = campaign.get("cases")
            require(isinstance(campaign_cases, list) and len(campaign_cases) == len(CAMPAIGN_CASES),
                    "campaign requires exactly five cases")
            previous_end = start
            for number_, (case, (obj, destination)) in enumerate(zip(campaign_cases, CAMPAIGN_CASES), 1):
                def check_campaign_case():
                    nonlocal previous_end
                    case_identity(case, (obj, destination))
                    require(all(case.get(key) is True for key in
                                ("pass", "reset_confirmed", "engine_matches_every_sample", "pick_trace_confirmed"))
                            and case.get("errors") == [] and case.get("observer_errors") == []
                            and case.get("uncertain_order") is False, "campaign case failed/incomplete")
                    directory = campaign_path.parent / f"case-{number_}-{obj}"
                    require(case == inputs.json(directory / "receipt.json"), "campaign case differs from saved receipt")
                    case_start = number(case.get("started_at"), "case started_at")
                    case_end = number(case.get("finished_at"), "case finished_at")
                    require(previous_end <= case_start < case_end <= end, "campaign case intervals overlap/outside campaign")
                    previous_end = case_end
                    times = full_audit(inputs, case, directory, obj, destination)
                    require(case_start <= times[0] < times[-1] <= case_end, "observer phases outside case")
                    order_ids = {}
                    bounded = list(times)
                    for phase, required, filename, index in (("pick", "pick_and_place", "01-pick.json", 0),
                                                            ("reset", "reset_scene", "02-reset.json", 2)):
                        identifier, turn_start, turn_end = visitor_turn(inputs, directory / filename, case.get(phase), required, ids)
                        require(times[index] <= turn_start < turn_end <= times[index + 1], "visitor turn outside observed phase")
                        bounded[index:index + 2] = [turn_start, turn_end]
                        order_ids[phase] = identifier
                    saved_rows = inputs.json(directory / "trace-evidence.json")
                    require(isinstance(saved_rows, list) and saved_rows
                            and saved_rows == [row for row in rows if case_start <= row["t"] <= case_end],
                            "saved case trace differs from bounded owner trace")
                    return {**native_outcome(saved_rows, bounded, obj, destination, case_interval=(case_start, case_end)), "order_ids": order_ids,
                            "phase_unix": dict(zip(PHASES, times)), "full_physical_audit_pass": True}
                case_result(report["campaign_native_cases"], obj, destination, check_campaign_case)
    except (OSError, ValueError, TypeError, KeyError) as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    try:
        inputs.unchanged()
    except (OSError, ValueError) as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    report["input_sha256"] = inputs.hashes()
    items = report["proof_native_cases"] + report["campaign_native_cases"]
    report["pass"] = (not report["errors"] and len(report["proof_native_cases"]) == 2
                      and (campaign_path is None or len(report["campaign_native_cases"]) == 5)
                      and all(item["pass"] is True for item in items))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--evidence-root", type=Path,
                        help="Archive root replacing process.repo; proof/campaign paths must be inside it")
    parser.add_argument("--output", type=Path, required=True, help="NEW report file; existing evidence is never overwritten")
    args = parser.parse_args(argv)
    report = validate(args.proof, args.campaign, evidence_root=args.evidence_root)
    try:
        require(str(args.output.resolve()) not in report["input_sha256"], "output overlaps input evidence")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError) as error:
        print(json.dumps({"pass": False, "output_error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({"pass": report["pass"], "proof_cases": len(report["proof_native_cases"]),
                      "campaign_cases": len(report["campaign_native_cases"]), "output": str(args.output)}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
