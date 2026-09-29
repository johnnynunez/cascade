#!/usr/bin/env python3
"""Read-only Spark receipt, live ownership and health checks with compact output."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


CASES = (("green cube", "green square", "green_cube"), ("orange", "open box", "orange"))
CAMERAS = {"cam0", "side", "proof"}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def verify(repo: Path, expected_ref: str) -> dict:
    require(re.fullmatch(r"[0-9a-f]{40}", expected_ref) is not None,
            "--expected-ref must be the full pinned commit")
    repo = repo.resolve(strict=True)
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", PYTHONDONTWRITEBYTECODE="1")

    def read(name: str) -> dict:
        return json.loads((repo / name).read_text())

    def run(args: list[str], label: str, *, child_env=None) -> str:
        try:
            result = subprocess.run(args, cwd=repo, env=child_env or env,
                                    capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            raise ValueError(label + " timed out") from None
        require(result.returncode == 0, label + " failed; private output withheld")
        return result.stdout

    install = read("runs/.install/install.json")
    desktop = read("runs/.install/desktop-latest.json")
    proof_path = repo / "runs/.launch/profile-cascade-demo/proof.json"
    proof = json.loads(proof_path.read_text())
    require(install["repo"] == str(repo) and install["source_commit"] == expected_ref,
            "installation path or pin does not match")
    require(install["source_dirty"] is False and install["profile"] == "spark" and
            install["brain"] == "qwen" and install["eula_accepted"] is True,
            "installation receipt is not a clean, consented Spark Qwen setup")
    require(run(["git", "rev-parse", "HEAD"], "checkout commit").strip() == expected_ref,
            "checkout commit does not match")
    require(not run(["git", "status", "--porcelain"], "checkout status").strip(),
            "checkout has source changes")
    require(desktop["repo"] == str(repo) and desktop["action"] == "launch" and
            desktop["exit_code"] == 0 and type(desktop["attached"]) is bool,
            "launch receipt does not describe a successful launch")
    require(proof["verified"] is True and proof["sim"] == "isaac" and
            isinstance(proof["model"], str) and proof["model"].endswith("/Qwen/Qwen3.8-27B"),
            "physical proof or model does not match")
    require(desktop["proof"]["path"] == str(proof_path) and
            all(desktop["proof"][key] == proof[key] for key in ("session_id", "started_at", "process")),
            "proof identity does not match the launch receipt")
    timestamps = (desktop["started_at"], desktop["finished_at"], proof["started_at"])
    require(all(type(value) in (int, float) and math.isfinite(value) for value in timestamps),
            "proof timestamps are invalid")
    start, finish, proven = timestamps
    require(start <= finish and proven <= finish and (desktop["attached"] or start <= proven),
            "proof is stale or outside the launch interval")
    require(isinstance(proof["cases"], list) and len(proof["cases"]) == len(CASES),
            "proof must contain exactly two placement cases")
    cases = {(case["object"], case["destination"]): case for case in proof["cases"]}
    require(set(cases) == {(obj, target) for obj, target, _ in CASES},
            "proof cases are missing, duplicated or unexpected")
    summary = []
    for obj, target, prop in CASES:
        case = cases[(obj, target)]
        physics = case["physics"]
        cameras = physics["event_cameras"]
        require(physics["pass"] is True and cameras["pass"] is True and prop in case["props_reset"] and
                set(cameras["cameras"]) == CAMERAS and
                all(cameras["cameras"][name]["pass"] is True for name in CAMERAS),
                "placement, event cameras or reset failed: " + obj)
        summary.append({"object": obj, "destination": target,
                        "physics": True, "event_cameras": True, "reset": True})
    # Scope the health request to the installed demo, even in a personal shell.
    health_env = {key: value for key, value in env.items() if not key.startswith("OPENCLAW_")}
    health_env.update(OPENCLAW_STATE_DIR=str(proof_path.parent / "openclaw"),
                      OPENCLAW_CONFIG_PATH=str(proof_path.parent / "openclaw/openclaw.json"))
    health = json.loads(run([str(repo / ".openclaw-cli/bin/openclaw"), "--profile", "cascade-demo",
                             "health", "--json"], "OpenClaw health", child_env=health_env))
    require(health["ok"] is True, "OpenClaw health is not ready")
    run([sys.executable, "-B", str(repo / "scripts/spark_browser.py"),
         "--repo", str(repo), "--check"], "live process, chat and camera binding")
    return {"repo": str(repo), "source_commit": expected_ref, "source_dirty": False,
            "profile": "spark", "brain": "qwen", "eula_accepted": True, "health": True,
            "live_binding": True, "attached": desktop["attached"], "verified": True,
            "sim": "isaac", "model": proof["model"], "cases": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-ref", required=True)
    args = parser.parse_args()
    try:
        result = verify(args.repo, args.expected_ref)
    except (OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        # Do not echo malformed file contents or captured backend responses.
        reason = str(exc) if type(exc) is ValueError else "missing or invalid verification data"
        print("NOT READY: " + reason, file=sys.stderr)
        return 1
    print("READY " + json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
