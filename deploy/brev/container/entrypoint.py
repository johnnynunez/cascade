#!/usr/bin/env python3
"""Prepare persistent runtime paths and run the existing owned demo supervisor."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def symlink(path, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() and path.resolve() == target.resolve():
        return
    if path.exists() or path.is_symlink():
        raise ValueError(f"Unexpected existing runtime path: {path.name}")
    path.symlink_to(target, target_is_directory=target.is_dir())


def prepare():
    os.umask(0o077)
    root = Path("/workspace")
    data = Path(os.environ.get("PAAI_DATA_DIR", "/data"))
    if os.path.lexists(data / "acceptance/STOP") or os.path.lexists(root / "STOP"):
        raise InterruptedError("The retained STOP marker prevents runtime startup")
    if not (root / "src/cascade").is_dir():
        raise ValueError("The prepared CASCADE source mount is missing")
    if os.environ.get("ACCEPT_EULA") != "Y":
        raise ValueError("The Isaac EULA must be explicitly accepted in deployment configuration")
    for name in ("state", "private/openclaw", "home", "logs/brain", "web", "cache/kit",
                 "cache/cuda", "cache/warp", "cache/shaders", "recordings", "runs"):
        (data / name).mkdir(mode=0o700, parents=True, exist_ok=True)
    symlink(root / ".venv", Path("/opt/paai/venv"))
    symlink(root / "runtime/openclaw", Path("/opt/paai/openclaw"))
    symlink(root / "runs", data / "runs")
    upstream = Path("/opt/paai/openclaw/node_modules/openclaw/dist/control-ui")
    ui = Path(os.environ["PAAI_OPENCLAW_UI_ROOT"])
    version = json.loads((upstream.parent.parent / "package.json").read_text())["version"]
    patches = root / "web/openclaw-ui"
    materialized = subprocess.run([os.environ.get("PAAI_NODE", "/usr/local/bin/node"),
                                  "/opt/paai/materialize_ui.mjs", str(upstream), str(ui),
                                  str(patches), version], capture_output=True, text=True, timeout=90)
    if materialized.returncode:
        raise RuntimeError("Native UI materialization failed")
    bind = os.environ["PAAI_HTTP_BIND"]
    origin = f"http://{bind}:8092"
    urls = {"tutorial": origin + "/guide/", "openclaw": origin + "/openclaw/",
            "cameras": origin + "/cameras/"}
    urls_path = data / "state/urls.json"
    urls_path.write_text(json.dumps(urls, indent=2) + "\n")
    return root, urls_path


def setup_failure_diagnostic(output):
    allowed_errors = {
        "ValueError", "RuntimeError", "FileNotFoundError", "PermissionError",
        "TimeoutError", "TimeoutExpired", "ModuleNotFoundError", "ImportError",
        "OSError", "KeyError", "TypeError", "JSONDecodeError", "UnicodeDecodeError",
    }
    for line in reversed(output[-8192:].splitlines()[-16:]):
        try:
            record = json.loads(line)
        except (ValueError, TypeError):
            continue
        status = record.get("status") if isinstance(record, dict) else None
        if not isinstance(status, str) or status not in {"FAILED", "FAIL", "STOPPED"}:
            continue
        error = record.get("error")
        error = error if isinstance(error, str) and error in allowed_errors else "unknown"
        return f"status={record['status']}, error={error}"
    return "status=unknown, error=unknown"


def main():
    root, urls = prepare()
    runtime = root / "deploy/runtime/runtime.py"
    setup = subprocess.run([sys.executable, str(runtime), "setup", "--root", str(root),
                            "--urls-file", str(urls)], capture_output=True, text=True, timeout=120)
    if setup.returncode:
        # Configuration CLI output can contain authentication material.
        raise RuntimeError("Runtime setup failed (" + setup_failure_diagnostic(setup.stdout)
                           + "); private configuration output withheld")
    os.execv(sys.executable, [sys.executable, str(runtime), "serve", "--root", str(root),
                             "--timeout-s", "1200"])


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        stopped = isinstance(error, InterruptedError)
        print(json.dumps({"status": "STOPPED" if stopped else "FAILED", "reason": str(error)}), flush=True)
        raise SystemExit(0 if stopped else 1)
