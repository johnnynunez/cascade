"""Spark verification is read-only and rejects invalid receipts and live state."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/spark_verify.py"


@pytest.fixture
def verify_boundary(tmp_path):
    repo = tmp_path / "paai-spark"
    state = repo / "runs/.launch/profile-cascade-demo"
    install_dir = repo / "runs/.install"
    for directory in (state, install_dir, repo / "scripts", repo / ".openclaw-cli/bin", tmp_path / "bin"):
        directory.mkdir(parents=True, exist_ok=True)
    pin = "a" * 40
    install = {"repo": str(repo), "source_commit": pin, "source_dirty": False,
               "profile": "spark", "brain": "qwen", "eula_accepted": True,
               "large_inventory": "inventory-detail-" * 10000}
    proof = {"verified": True, "sim": "isaac", "model": "fixture/Qwen/Qwen3.8-27B",
             "started_at": 150, "session_id": "fixture-proof", "process": {"pid": 123, "owner": "fixture"},
             "cases": [{"object": obj, "destination": target, "props_reset": [prop], "physics": {
                 "pass": True, "event_cameras": {"pass": True, "cameras": {
                     camera: {"pass": True} for camera in ("cam0", "side", "proof")}},
                 "large_audit": "physics-detail-" * 10000}}
                 for obj, target, prop in (("green cube", "green square", "green_cube"), ("orange", "open box", "orange"))]}
    desktop = {"repo": str(repo), "action": "launch", "exit_code": 0,
               "attached": False, "started_at": 100, "finished_at": 200,
               "proof": {"path": str(state / "proof.json"), **{key: proof[key] for key in ("session_id", "started_at", "process")}}}
    receipts = {install_dir / "install.json": install,
                install_dir / "desktop-latest.json": desktop, state / "proof.json": proof}
    (repo / "scripts/spark_browser.py").write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "Path('live-check.args').write_text(json.dumps(sys.argv[1:]))\n"
        "raise SystemExit(int(os.environ['DOC_LIVE_EXIT']))\n"
    )
    executable = repo / ".openclaw-cli/bin/openclaw"
    executable.write_text(
        "#!/usr/bin/python3\nimport json,os,sys\nfrom pathlib import Path\n"
        "Path('health.args').write_text(json.dumps({'args':sys.argv[1:], 'state':os.environ.get('OPENCLAW_STATE_DIR')}))\n"
        "print(json.dumps({'ok':os.environ['DOC_HEALTH_OK']=='1', 'details':'private-health-detail-'*10000}))\n"
    )
    executable.chmod(0o755)
    git = tmp_path / "bin/git"
    git.write_text("#!/bin/sh\nif [ \"$1\" = status ]; then printf '%s' \"$VERIFY_GIT_STATUS\"; else printf '%s\\n' '" + pin + "'; fi\n")
    git.chmod(0o755)
    env = {"HOME": str(tmp_path), "PATH": str(tmp_path / "bin") + os.pathsep + os.defpath,
           "VERIFY_GIT_STATUS": "", "DOC_LIVE_EXIT": "0", "DOC_HEALTH_OK": "1"}

    def run():
        for path, record in receipts.items():
            path.write_text(json.dumps(record))
        before = {path: path.read_bytes() for path in receipts}
        result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--repo", str(repo), "--expected-ref", pin],
                                cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
        assert before == {path: path.read_bytes() for path in receipts}
        assert not list(repo.rglob("__pycache__"))
        return result

    return {"repo": repo, "home": tmp_path, "state": state, "install": install,
            "proof": proof, "desktop": desktop, "env": env, "run": run}


@pytest.mark.parametrize("attached", [False, True])
def test_verifier_is_concise_and_validates_fresh_or_attached_proof(verify_boundary, attached):
    fixture = verify_boundary
    fixture["desktop"]["attached"] = attached
    if attached:
        fixture["desktop"]["started_at"] = 180
    result = fixture["run"]()
    assert result.returncode == 0, result.stderr
    assert len(result.stdout) < 2000
    assert "private-health-detail" not in result.stdout
    assert "large-launch-detail" not in result.stdout
    assert "inventory-detail" not in result.stdout
    assert "physics-detail" not in result.stdout
    summary = json.loads(next(line.removeprefix("READY ") for line in result.stdout.splitlines() if line.startswith("READY ")))
    assert summary["attached"] is attached
    assert summary["health"] is True and summary["live_binding"] is True
    assert len(summary["cases"]) == 2
    assert all(case["physics"] and case["event_cameras"] and case["reset"] for case in summary["cases"])
    health = json.loads((fixture["repo"] / "health.args").read_text())
    assert health == {"args": ["--profile", "cascade-demo", "health", "--json"],
                      "state": str(fixture["state"] / "openclaw")}
    assert json.loads((fixture["repo"] / "live-check.args").read_text()) == ["--repo", str(fixture["repo"]), "--check"]


@pytest.mark.parametrize("defect", ["path", "pin", "dirty", "current_dirty", "eula", "model", "stale", "binding",
                                    "physics", "event_cameras", "one_camera", "reset", "health", "live",
                                    "duplicate_case", "extra_case", "missing_case", "missing_camera", "unverified", "sim"])
def test_verifier_never_prints_ready_after_failed_evidence(verify_boundary, defect):
    fixture = verify_boundary
    if defect == "current_dirty":
        fixture["env"]["VERIFY_GIT_STATUS"] = " M scripts/desktop.py\n"
    elif defect == "duplicate_case":
        fixture["proof"]["cases"][1] = fixture["proof"]["cases"][0]
    elif defect == "extra_case":
        fixture["proof"]["cases"].append(fixture["proof"]["cases"][0])
    elif defect == "missing_case":
        fixture["proof"]["cases"].pop()
    elif defect == "missing_camera":
        fixture["proof"]["cases"][0]["physics"]["event_cameras"]["cameras"].pop("cam0")
    elif defect == "unverified":
        fixture["proof"]["verified"] = False
    elif defect == "sim":
        fixture["proof"]["sim"] = "other"
    elif defect == "path":
        fixture["install"]["repo"] = "/wrong/checkout"
    elif defect == "pin":
        fixture["install"]["source_commit"] = "0" * 40
    elif defect == "dirty":
        fixture["install"]["source_dirty"] = True
    elif defect == "eula":
        fixture["install"]["eula_accepted"] = False
    elif defect == "model":
        fixture["proof"]["model"] = "fixture/another-model"
    elif defect == "stale":
        fixture["desktop"]["started_at"] = 180
    elif defect == "binding":
        fixture["desktop"]["proof"]["session_id"] = "another-session"
    elif defect == "physics":
        fixture["proof"]["cases"][0]["physics"]["pass"] = False
    elif defect == "event_cameras":
        fixture["proof"]["cases"][1]["physics"]["event_cameras"]["pass"] = False
    elif defect == "one_camera":
        fixture["proof"]["cases"][1]["physics"]["event_cameras"]["cameras"]["side"]["pass"] = False
    elif defect == "reset":
        fixture["proof"]["cases"][1]["props_reset"] = []
    elif defect == "health":
        fixture["env"]["DOC_HEALTH_OK"] = "0"
    elif defect == "live":
        fixture["env"]["DOC_LIVE_EXIT"] = "1"
    result = fixture["run"]()
    assert result.returncode != 0
    assert not any(line.startswith("READY ") for line in result.stdout.splitlines())


