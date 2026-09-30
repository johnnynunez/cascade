#!/usr/bin/env python3
"""External validation tool: five native attendee orders, unchanged passive audit.

Run only after the normal desktop launcher reaches READY, with no other visitor
using the demo. This script submits natural language to the existing visitor API;
it never constructs a robot runtime, starts services, or changes proof.json,
installation receipts, configuration, physics, or acceptance thresholds.

Example:
  .venv/bin/python ../spark-clean-native-campaign.py --repo "$PWD" \
    --expected-ref <full-commit> --output runs/native-ui-five-props-01

An uncertain POST/order is never retried and never followed by a competing reset.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


CASES = (
    ("green_cube", "green square"), ("orange", "open box"),
    ("pink_cube", "green square"), ("lemon", "open box"),
    ("tomato_can", "green square"),
)
READ_TOOLS = {"describe_scene", "localize_object", "camera_snapshot", "world_state"}
ORIGIN = "http://127.0.0.1:8092"


class UncertainOrder(RuntimeError):
    """The submitted action may still be running; stop issuing orders."""


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def http_json(method, path, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(ORIGIN + path, data=body, method=method,
                      headers={"Accept": "application/json", "Content-Type": "application/json",
                               "Origin": ORIGIN})
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=12) as reply:
            data = reply.read(1024 * 1024 + 1)
            if len(data) > 1024 * 1024:
                raise ValueError("Visitor response exceeded 1 MiB")
            result = json.loads(data)
    except HTTPError as error:
        raise RuntimeError(f"Visitor HTTP {error.code}; no request retry") from None
    if not isinstance(result, dict):
        raise ValueError("Visitor response must be an object")
    return result


class VisitorClient:
    def __init__(self, guard, *, transport=http_json, poll_interval=1.0, timeout=330):
        self.guard, self.transport = guard, transport
        self.poll_interval, self.timeout = poll_interval, timeout

    def turn(self, message, required_tool, path):
        self.guard()
        initial = self.transport("GET", "/api/chat")
        if (initial.get("enabled") is not True or initial.get("ready") is not True
                or initial.get("busy") is not False or initial.get("agent") != "cascade-demo"):
            raise RuntimeError("Attendee API is not READY and idle; no order submitted")
        record = {"message": message, "required_tool": required_tool,
                  "started_at": time.time(), "initial": initial, "status": "submitting"}
        write_json(path, record)
        try:
            # Exactly one POST. Even a lost response may mean the action started.
            accepted = self.transport("POST", "/api/chat", {"message": message})
            identifier = accepted.get("id")
            if (accepted.get("status") != "running" or not isinstance(identifier, str)
                    or re.fullmatch(r"[a-f0-9]{32}", identifier) is None):
                raise ValueError("Visitor returned an invalid order acknowledgment")
            record.update(status="running", accepted=accepted)
            write_json(path, record)
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                self.guard()
                current = self.transport("GET", "/api/chat?id=" + identifier)
                record["last_response"] = current
                write_json(path, record)
                order = current.get("order") or {}
                if order.get("id") != identifier:
                    raise ValueError("Visitor lost or replaced the submitted order")
                if order.get("status") == "done" and current.get("busy") is False:
                    record.update(status="done", finished_at=time.time())
                    write_json(path, record)
                    break
                if order.get("status") in ("error", "uncertain"):
                    raise ValueError("Visitor cannot confirm that the order finished")
                time.sleep(self.poll_interval)
            else:
                raise TimeoutError("Attendee order did not finish within the observation budget")
        except Exception as error:
            record.update(status="uncertain", error=f"{type(error).__name__}: {error}")
            write_json(path, record)
            raise UncertainOrder(record["error"]) from error
        names = order.get("tools")
        if (not isinstance(names, list) or required_tool not in names
                or not set(names) <= READ_TOOLS | {required_tool}
                or type(order.get("tool_failures")) is not int or order["tool_failures"] != 0):
            raise RuntimeError("Completed attendee turn failed or used unexpected native tools")
        return record


def source_hashes(repo):
    names = subprocess.check_output([
        "git", "--no-optional-locks", "-C", str(repo), "ls-files", "-z", "--",
        "src", "scripts", "configs", "demo/kitchen", "demo/scene", "demo/scene_identity.py",
        "deploy/brev/visitor.py", "deploy/brev/visitor_chat.py",
    ]).decode().split("\0")
    suffixes = {".py", ".sh", ".yaml", ".yml", ".json", ".usda", ".urdf"}
    return {name: digest(repo / name) for name in names if name and Path(name).suffix in suffixes}


def run_case(api, witness_type, validators, binding, directory, object_name, destination):
    directory.mkdir()
    trace = Path(binding["trace"])
    target = object_name.replace("_", " ")
    receipt = {"object": object_name, "destination": destination, "pass": False,
               "started_at": time.time(), "errors": [], "reset_confirmed": False}
    path = directory / "receipt.json"
    observer = witness_type(directory / "physics", object_name, destination)
    pick_started = reset_started = None
    uncertain = False
    try:
        with observer:
            if {row["physics"].get("engine") for row in observer.records} != {"physx"}:
                raise RuntimeError("This campaign requires the unchanged default PhysX engine")
            observer.mark("pick_begin")
            pick_started = time.time()
            try:
                receipt["pick"] = api.turn(f"Please put the {target} in the {destination}.",
                                           "pick_and_place", directory / "01-pick.json")
            except UncertainOrder:
                uncertain = True
                raise
            except Exception:
                receipt["errors"].append(traceback.format_exc())
            # A completed but failed pick still gets the normal native reset.
            try:
                observer.settle(wall_timeout=90)
                observer.capture_frames("placed")
            except Exception:
                receipt["errors"].append(traceback.format_exc())
            observer.mark("pick_end")
            observer.mark("reset_begin")
            reset_started = time.time()
            receipt["reset"] = api.turn("Let's start over.", "reset_scene", directory / "02-reset.json")
            receipt["props_reset"] = validators.validate_reset_trace(trace, reset_started,
                                                                       "isaac", target)
            receipt["reset_confirmed"] = True
            receipt["world"] = api.turn("What is the session status?", "world_state", directory / "03-world.json")
            receipt["inspection"] = api.turn("What can you see on the table?", "describe_scene",
                                             directory / "04-inspect.json")
            observer.settle(wall_timeout=90)
            observer.mark("reset_end")
            observer.capture_frames("reset")
    except UncertainOrder:
        uncertain = True
        receipt["errors"].append(traceback.format_exc())
    except Exception:
        receipt["errors"].append(traceback.format_exc())
    finally:
        if observer.out.exists():
            try:
                receipt["physics"] = observer.audit()
                receipt["engine_matches_every_sample"] = bool(observer.records) and all(
                    row["physics"].get("engine") == "physx" for row in observer.records)
                if pick_started is not None:
                    validators.validate_pick_trace(trace, pick_started, target,
                                                   placement_audit=receipt["physics"])
                    receipt["pick_trace_confirmed"] = True
            except Exception:
                receipt["errors"].append(traceback.format_exc())
        try:
            rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
            current = [row for row in rows if row.get("t", 0) >= receipt["started_at"]]
            write_json(directory / "trace-evidence.json", current)
        except Exception:
            receipt["errors"].append(traceback.format_exc())
        receipt.update(finished_at=time.time(), observer_errors=list(observer.errors),
                       uncertain_order=uncertain, samples=len(observer.records))
        receipt["pass"] = bool(not receipt["errors"] and receipt.get("physics", {}).get("pass")
                               and receipt.get("pick_trace_confirmed") and receipt["reset_confirmed"]
                               and receipt.get("engine_matches_every_sample"))
        write_json(path, receipt)
    return receipt


def run(args):
    repo = args.repo.resolve(strict=True)
    output = args.output if args.output.is_absolute() else repo / args.output
    output = output.resolve()
    if not output.is_relative_to(repo / "runs") or output.exists():
        raise ValueError("--output must be a NEW evidence directory beneath this checkout's runs/")
    if output.is_relative_to(repo / "runs/.launch") or output.is_relative_to(repo / "runs/.install"):
        raise ValueError("Evidence must not overlap launch or installation receipts")
    if re.fullmatch(r"[a-f0-9]{40}", args.expected_ref) is None:
        raise ValueError("--expected-ref requires a full commit hash")
    sys.path[:0] = [str(repo / "src"), str(repo / "scripts")]
    from cascade.apps.process_owner import is_live, live_records, load_owner

    validators = load_module("external_native_validators", repo / "scripts/demo_proof.py")
    passive = load_module("external_native_passive", repo / "demo/kitchen/physics/spark_proof.py")
    scene = passive.shared.load_expected_scene_geometry(repo / "demo/scene/kitchen_config.json")

    class Witness(passive.SparkKitchenWitness):
        def __init__(self, out, object_name, destination):
            # Only extend the constructor's two-case routing list. Snapshot,
            # camera audit, physical tolerances and geometry stay inherited.
            passive.shared.GpuProofObserver.__init__(
                self, out, port=8611, interval_s=.15, budget_s=900,
                object_name=object_name, destination_name=destination,
                expected_scene_geometry=scene)

    state = repo / "runs/.launch/profile-cascade-demo"
    protected = [repo / "runs/.install/install.json", repo / "runs/.install/desktop-latest.json",
                 state / "proof.json"]
    before = {str(path.relative_to(repo)): digest(path) for path in protected}
    proof = json.loads((state / "proof.json").read_text())
    owner = load_owner(state, repo, "cascade-demo")
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "--no-optional-locks", "-C", str(repo),
                                     "status", "--porcelain"], text=True).strip()
    if commit != args.expected_ref or dirty or proof.get("verified") is not True or proof.get("sim") != "isaac":
        raise RuntimeError("Expected clean source commit and a current verified Isaac proof")
    if owner is None or not is_live(proof.get("process"), owner):
        raise RuntimeError("The verified proof MCP is no longer live")
    binding = {key: proof[key] for key in ("session_id", "model", "process", "trace")}
    expected_trace = Path(proof["process"]["run_dir"]).resolve() / "trace.jsonl"
    if Path(binding["trace"]).resolve() != expected_trace or expected_trace.parent.parent != repo / "runs":
        raise RuntimeError("Proof trace does not belong to its owner-bound MCP")

    def guard():
        if any(digest(repo / name) != value for name, value in before.items()):
            raise RuntimeError("Protected installation/launch/proof receipt changed during campaign")
        roles = {item["role"] for item in live_records(state, owner)}
        if not is_live(binding["process"], owner) or not {"qwen", "isaac_bridge", "gateway_child", "visitor", "cameras"} <= roles:
            raise RuntimeError("An owner-bound demo process changed or stopped")

    guard()
    output.mkdir(parents=True)
    hashes = source_hashes(repo)
    report = {"schema_version": 1, "scope": "External native visitor campaign; not product code or a new READY receipt",
              "repo": str(repo), "source_commit": commit, "source_sha256": hashes,
              "harness_sha256": digest(__file__), "protected_receipts_sha256": before,
              "binding": binding, "engine": "physx", "planned_cases": [list(pair) for pair in CASES],
              "started_at": time.time(), "run_complete": False, "pass": False, "cases": [], "errors": []}
    checkpoint = output / "campaign.json"
    write_json(checkpoint, report)
    api = VisitorClient(guard)
    try:
        report["initial_inspection"] = api.turn("What can you see on the table?", "describe_scene",
                                                output / "initial-inspection.json")
        for number, (object_name, destination) in enumerate(CASES, 1):
            guard()
            print(f"BEGIN {number}/5 {object_name} -> {destination}", flush=True)
            case = run_case(api, Witness, validators, binding, output / f"case-{number}-{object_name}",
                            object_name, destination)
            report["cases"].append(case)
            write_json(checkpoint, report)
            print(f"{'PASS' if case['pass'] else 'FAIL'} {object_name}; reset={case['reset_confirmed']}", flush=True)
            if case["uncertain_order"] or not case["reset_confirmed"]:
                raise RuntimeError("Stopped: order/reset is unconfirmed; no further motion requests")
            if args.fail_fast and not case["pass"]:
                break
        report["run_complete"] = len(report["cases"]) == len(CASES)
    except BaseException:
        report["errors"].append(traceback.format_exc())
    finally:
        try:
            guard()
            report["protected_receipts_unchanged"] = True
            report["source_unchanged"] = source_hashes(repo) == hashes
        except Exception:
            report["errors"].append(traceback.format_exc())
        report["finished_at"] = time.time()
        report["pass"] = bool(report["run_complete"] and not report["errors"]
                              and report.get("source_unchanged") and report.get("protected_receipts_unchanged")
                              and all(case["pass"] for case in report["cases"]))
        write_json(checkpoint, report)
    print(json.dumps({"pass": report["pass"], "cases": len(report["cases"]),
                      "passed": sum(case["pass"] for case in report["cases"]), "evidence": str(checkpoint)}), flush=True)
    return 0 if report["pass"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    # This lock excludes another copy of this external harness, not the visitor
    # itself. Every order still passes through the visitor's own admission lock.
    with open("/tmp/cascade-native-visitor-campaign-8092.lock", "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
