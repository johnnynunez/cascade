#!/usr/bin/env python3
"""Linux desktop front door for the existing Spark installer and proof launcher.

Registration starts nothing and accepts no license. Terminal windows show real
progress; full logs persist even when installation or physical proof fails.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from install_support import EULA_URL, eula_accepted


def _exec_arg(value: str) -> str:
    # Desktop Entry value escaping, then Exec quoting (not shell quoting).
    if any(c in value for c in "\n\r\x00"):
        raise ValueError("desktop paths cannot contain line breaks or NUL")
    value = value.replace("\\", "\\\\\\\\").replace('"', '\\\\"')
    value = value.replace("$", "\\\\$").replace("`", "\\\\`").replace("%", "%%")
    return '"' + value + '"'


def register(repo: Path, *, desktop_dir: Path | None = None) -> list[Path]:
    """Install checkout-specific entries, leaving every other app untouched."""
    repo = repo.resolve()
    if not (repo / "scripts/desktop.py").is_file():
        raise RuntimeError(f"not a CASCADE checkout: {repo}")
    data = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    destinations = [data / "applications"]
    if desktop_dir is None:
        if shutil.which("xdg-user-dir"):
            result = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True, check=True)
            desktop_dir = Path(result.stdout.strip()) if result.stdout.strip() else None
        elif (Path.home() / "Desktop").is_dir():
            desktop_dir = Path.home() / "Desktop"
    # xdg-user-dir returns HOME when the desktop directory was disabled.
    if desktop_dir is not None and desktop_dir != Path.home():
        destinations.append(desktop_dir)
    identifier = hashlib.sha256(str(repo).encode()).hexdigest()[:12]
    paths = []
    for action, name in (("install", "Install CASCADE (Spark)"), ("launch", "CASCADE (Spark)")):
        text = ("[Desktop Entry]\nVersion=1.0\nType=Application\n"
                f"Name={name}\nComment=Isaac Sim + local model + isolated OpenClaw; physical proof required\n"
                f"Exec=/usr/bin/python3 {_exec_arg(str(repo / 'scripts/desktop.py'))} {action} --repo {_exec_arg(str(repo))}\n"
                "Icon=applications-engineering\nTerminal=true\nStartupNotify=false\nCategories=Development;Science;\n")
        for directory in destinations:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"cascade-{action}-{identifier}.desktop"
            if not path.is_file() or path.read_text() != text:
                temporary = path.with_suffix(f".{os.getpid()}.tmp")
                temporary.write_text(text)
                temporary.chmod(0o755)
                temporary.replace(path)
            paths.append(path)
    return paths


def confirm_eula() -> bool:
    message = ("Install CASCADE for this Linux Spark using Isaac Sim 6.1, the local model and OpenClaw.\n\n"
               f"Review the NVIDIA Isaac Sim / Omniverse EULA:\n{EULA_URL}\n\n"
               "Do you explicitly agree to this EULA? Downloads may be large. No driver or OS changes. "
               "If installation succeeds, the simulation proof will move the simulated robot. No physical hardware.")
    if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) and shutil.which("zenity"):
        return subprocess.run(["zenity", "--question", "--title=Install CASCADE", "--no-markup",
                               "--width=620", "--ok-label=I agree and install", "--cancel-label=Cancel",
                               "--default-cancel", "--text=" + message]).returncode == 0
    if sys.stdin.isatty():
        print(message, flush=True)
        return input("Type 'I agree' to install, or Enter to cancel: ").strip() == "I agree"
    return False


def _write_status(path: Path, report: dict) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def perform(repo: Path, action: str, *, prepare_only: bool = False, dry_run: bool = False) -> int:
    repo = repo.resolve()
    if action not in ("install", "launch"):
        raise ValueError(f"unknown desktop action: {action}")
    if prepare_only and action != "install":
        raise ValueError("--prepare-only belongs to install, not launch")
    env = os.environ.copy()
    env.update(CASCADE_INSTALL_PROFILE="spark", CASCADE_OPENCLAW_PROFILE="cascade-demo",
               PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    if action == "install":
        command = ["bash", str(repo / "scripts/install.sh"), "--dir", str(repo), "--profile", "spark", "--accept-eula"]
        if prepare_only:
            command.append("--prepare-only")
    else:
        command = [str(repo / ".venv/bin/python"), str(repo / "scripts/install_support.py"),
                   "launch", "--repo", str(repo), "--profile", "spark", "--brain", "qwen"]
    if dry_run:
        print(json.dumps({"action": action, "command_after_consent": command,
                          "dry_run": True, "services_started": False}))
        return 0
    if action == "install":
        if not confirm_eula():
            print("[desktop] Cancelled: no installation, services or license acceptance.", flush=True)
            return 2
    else:
        if not eula_accepted(repo):
            raise RuntimeError("No explicit license-consent receipt for this checkout. Open 'Install CASCADE (Spark)' first.")
        record = json.loads((repo / "runs/.install/install.json").read_text())
        # Existing Spark consent remains valid across the model installation fix.
        if record.get("profile") != "spark" or record.get("brain") not in ("qwen", "cosmos"):
            raise RuntimeError("This is not a prepared Spark installation. Open 'Install CASCADE (Spark)'.")
        if not os.access(repo / ".venv/bin/python", os.X_OK):
            raise RuntimeError("Application environment is missing. Open 'Install CASCADE (Spark)' to repair it; no fallback.")
        for key, value in record.get("isaac_environment", {}).items():
            if key in ("ISAACSIM_PATH", "ISAACSIM_PYTHON_EXE"):
                env[key] = value
    state = repo / "runs/.install"
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (state / "desktop.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"[desktop] An install/launch is already in progress. See {state / 'desktop-latest.json'}", flush=True)
            return 75
        directory = state / ("desktop-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
        directory.mkdir(mode=0o700)
        log_path = directory / "progress.log"
        report = {"action": action, "repo": str(repo), "log": str(log_path), "started_at": time.time(), "exit_code": None}
        latest = state / "desktop-latest.json"
        _write_status(latest, report)
        print(f"[desktop] {action.upper()}: Isaac + local model + OpenClaw profile cascade-demo", flush=True)
        print(f"[desktop] Full progress log: {log_path}", flush=True)
        code = 1
        with log_path.open("w") as log:
            log_path.chmod(0o600)
            process = None
            try:
                process = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except KeyboardInterrupt:
                # Only this controller's private group, never borrowed services.
                code = 130
                if process is not None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=30)
            except OSError as exc:
                log.write(f"[desktop] ERROR: {exc}\n")
                raise
            finally:
                if process is not None and process.stdout is not None:
                    process.stdout.close()
                report.update(exit_code=code, finished_at=time.time())
                _write_status(latest, report)
        print(f"[desktop] Exit {code}. Log: {log_path}", flush=True)
        return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["register", "install", "launch"])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--desktop-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    def interrupted(_signum, _frame):
        # Terminal close and service-manager termination must use the same
        # owned-child cleanup as Ctrl-C, not abandon the private session.
        raise KeyboardInterrupt

    if args.action != "register" and not args.dry_run:
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGHUP, interrupted)
    code = 1
    try:
        if args.action == "register":
            if args.dry_run:
                print("[desktop] Would register install/launch entries. No writes or services.")
            else:
                print(json.dumps({"entries": [str(p) for p in register(args.repo, desktop_dir=args.desktop_dir)], "services_started": False}))
            return 0
        code = perform(args.repo, args.action, prepare_only=args.prepare_only, dry_run=args.dry_run)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"[desktop] ERROR: {exc}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        code = 130
    if code and sys.stdin.isatty():
        try:
            input("Press Enter to close this window. The diagnostic log is retained. ")
        except (EOFError, KeyboardInterrupt):
            pass
    return code


if __name__ == "__main__":
    raise SystemExit(main())
