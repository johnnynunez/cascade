"""Run Newton acceptance on CPU and persist honest, per-test JSON evidence.

    /tmp/cascade-newton-nhASTv/bin/python benchmark/diagnostics/run_newton_delivery.py

Arguments after -- are passed to pytest (e.g. -- -k box). All selected tests
are counted; setup/collection failures are retained. Test durations include
initialization/JIT/assertions and are NOT steady-state physics benchmarks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
import pytest
import warp as wp

REPO = Path(__file__).resolve().parents[2]


class Evidence:
    def __init__(self, output, invocation):
        self.output = output
        wp.init()
        self.report = {
            "schema_version": 1,
            "scope": "Newton CPU diagnostics, not a CASCADE or Isaac episode",
            "does_not_certify": ["Isaac Sim 6.1 binary/integration", "GPU/CUDA", "DGX Spark", "Cosmos", "CASCADE motion/safety/dispatcher", "camera rendering", "physical robot"],
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": invocation,
            "mutation": os.environ.get("NEWTON_DELIVERY_MUTATION"),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(), "machine": platform.machine(),
            "devices": [str(device) for device in wp.get_devices()],
            "versions": {name: importlib.metadata.version(name) for name in ("newton", "mujoco", "mujoco-warp", "warp-lang", "numpy", "pytest", "trimesh", "scipy")},
            "installed_packages": dict(sorted((distribution.metadata["Name"], distribution.version) for distribution in importlib.metadata.distributions())),
            "validation_source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__), Path(__file__).with_name("test_newton_delivery.py"))},
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "timing_note": "Pytest durations include JIT, construction, assertions and readback; not a throughput benchmark.",
            "collected": 0, "deselected": 0, "results": {}, "collection_errors": [], "warnings": [],
        }
        self.flush()

    def flush(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.output)

    def pytest_collection_finish(self, session):
        self.report["collected"] = len(session.items)
        for item in session.items:
            self.report["results"].setdefault(item.nodeid, {"nodeid": item.nodeid, "phases": [], "outcome": "not_run", "measurements": {}})
        self.flush()

    def pytest_deselected(self, items):
        self.report["deselected"] += len(items)

    def pytest_collectreport(self, report):
        if report.failed:
            self.report["collection_errors"].append(str(report.longrepr))
            self.flush()

    def pytest_warning_recorded(self, warning_message, when, nodeid, location):
        self.report["warnings"].append({"message": str(warning_message.message), "category": warning_message.category.__name__, "when": when, "nodeid": nodeid})
        self.flush()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        result = yield
        report = result.get_result()
        row = self.report["results"].setdefault(item.nodeid, {"nodeid": item.nodeid, "phases": [], "outcome": "not_run", "measurements": {}})
        row["phases"].append({"phase": report.when, "outcome": report.outcome, "duration_including_overhead_s": report.duration})
        row["measurements"] = getattr(item, "newton_measurements", {})
        if report.failed:
            row["outcome"] = "failed"
            row["failure"] = str(report.longrepr)
        elif report.skipped:
            row["outcome"] = "skipped"
        elif report.when == "call" and row["outcome"] == "not_run":
            row["outcome"] = "passed"
        self.flush()

    def pytest_sessionfinish(self, session, exitstatus):
        self.report["pytest_exit_code"] = int(exitstatus)
        self.report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        counts = Counter(row["outcome"] for row in self.report["results"].values())
        self.report["counts"] = {key: counts[key] for key in ("passed", "failed", "skipped", "not_run")}
        self.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mutation", choices=("disable_contacts", "zero_controls", "body_only_reset"),
                        help="Deliberate real-physics fault: acceptance assertions must fail")
    args, pytest_args = parser.parse_known_args()
    if args.mutation:
        os.environ["NEWTON_DELIVERY_MUTATION"] = args.mutation
    else:
        os.environ.pop("NEWTON_DELIVERY_MUTATION", None)
    if pytest_args[:1] == ["--"]:
        pytest_args.pop(0)
    directory = args.output_dir or REPO / "runs/newton-validation" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = directory.resolve()
    recorder = Evidence(directory / "metrics.json", [sys.executable, *sys.argv])
    result = pytest.main([str(Path(__file__).with_name("test_newton_delivery.py")), "-q", "--tb=short", "-o", "addopts=", "--junitxml=" + str(directory / "junit.xml"), *pytest_args], plugins=[recorder])
    # Read back and count the actual records, not the console's impression.
    readback = json.loads(recorder.output.read_text())
    counts = Counter(row["outcome"] for row in readback["results"].values())
    assert len(readback["results"]) == readback["collected"], "record count differs from collection"
    assert all(counts[key] == readback["counts"][key] for key in readback["counts"])
    print(json.dumps({"metrics_file": str(recorder.output), "collected": readback["collected"], "counts": readback["counts"], "deselected": readback["deselected"], "pytest_exit_code": int(result)}, indent=2))
    return int(result) or (1 if counts["skipped"] or counts["not_run"] or not readback["collected"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
