#!/usr/bin/env python3
"""Prepare models and the runtime image in a finite, supervised Brev job."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile

UNIT = "paai-demo-prepare.service"
DOCKER_SOCKET = "unix:///run/paai-demo-docker.sock"
IMAGE = "paai-demo:x86_64"
ACTIVE_STATES = {"activating", "active", "reloading", "deactivating"}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".prepare-",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_record(path):
    if not path.exists():
        return {}
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
        raise ValueError("Preparation receipt must be a small regular file")
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError("Preparation receipt must be a JSON object")
    return result


def capture(command):
    result = subprocess.run(command, capture_output=True, text=True,
                            stdin=subprocess.DEVNULL, timeout=20)
    return result.returncode, result.stdout.strip()


def status(root):
    record = read_record(root / "deployment/preparation.json")
    code, output = capture(["systemctl", "show", UNIT, "--property=ActiveState", "--value"])
    if code:
        raise RuntimeError("Cannot inspect supervised preparation")
    state = output
    return {"checked_at": stamp(), "unit": UNIT, "unit_state": state,
            "active": state in ACTIVE_STATES, "preparation": record}


def image_id():
    code, output = capture(["docker", "--host", DOCKER_SOCKET, "image", "inspect",
                            IMAGE, "--format", "{{.Id}}"])
    return output if code == 0 and re.fullmatch(r"sha256:[0-9a-f]{64}", output) else None


def run_logged(command, path, timeout):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
            raise
    if code:
        raise RuntimeError(f"Preparation command failed with exit {code}; private log: {path.name}")


def validate_plan(root, plan):
    if root.is_symlink() or not root.is_absolute() or root.resolve() != root:
        raise ValueError("Preparation root must be an absolute canonical directory")
    mount = Path(plan["storage_mount"])
    if (not mount.is_absolute() or not os.path.ismount(mount)
            or root == mount or not root.is_relative_to(mount)
            or root.stat().st_dev != mount.stat().st_dev
            or root.stat().st_dev == Path("/").stat().st_dev):
        raise ValueError("Preparation must remain on the admitted separate volume")
    for name in ("bundle_identity", "image_identity"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(plan.get(name, ""))):
            raise ValueError("Preparation identity is invalid")
    if not re.fullmatch(r"[0-9a-f]{40}", str(plan.get("source_revision", ""))):
        raise ValueError("Preparation source revision is invalid")
    if not isinstance(plan.get("source_files"), int) or plan["source_files"] <= 0:
        raise ValueError("Preparation source count is invalid")
    if plan.get("model_manifest", "profiles/qwen3.8-27b-q8.json") not in (
            "profiles/qwen3.8-27b-q8.json",):
        raise ValueError("Preparation model manifest is not an admitted Qwen quantization")


def prepare(root, plan):
    validate_plan(root, plan)
    deployment = root / "deployment"
    descriptor = os.open(deployment / ".prepare.lock", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another preparation worker owns this deployment") from None
        previous = read_record(deployment / "preparation.json")
        started = stamp()
        attempt = started.replace(":", "-")
        log_directory = root / "data/logs/preparation"
        log_directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        if previous:
            atomic(log_directory / f"previous-{attempt}.json", previous)
        record = {"status": "RUNNING", "started_at": started, "checked_at": started,
                  "bundle_identity": plan["bundle_identity"], "image_identity": plan["image_identity"],
                  "stage": "models", "model_download_command_exit": None,
                  "container_build_exit": None, "image_reused": False}

        def save():
            record["checked_at"] = stamp()
            atomic(deployment / "preparation.json", record)

        save()
        try:
            run_logged([sys.executable, str(Path(__file__).with_name("fetch_models.py")),
                        "--root", str(root), "--manifest",
                        str(Path(__file__).parent / plan.get("model_manifest", "profiles/qwen3.8-27b-q8.json"))],
                       log_directory / f"{attempt}-models.log", 7260)
            record["model_download_command_exit"] = 0
            record["stage"] = "image"
            save()
            prior_image = read_record(deployment / "image-build.json")
            current_image = image_id()
            if (prior_image.get("image_identity") == plan["image_identity"]
                    and current_image is not None and prior_image.get("image_id") == current_image):
                record["image_reused"] = True
            else:
                command = ["docker", "--host", DOCKER_SOCKET, "compose", "--project-name", "paai-demo",
                           "--env-file", str(deployment / "site.env"),
                           "-f", str(deployment / "compose.yaml"), "build"]
                run_logged(command, log_directory / f"{attempt}-build.log", 5400)
                current_image = image_id()
                if current_image is None:
                    raise RuntimeError("Build returned without a verifiable runtime image")
                atomic(deployment / "image-build.json", {"checked_at": stamp(),
                       "image_identity": plan["image_identity"], "image_id": current_image})
            record.update(container_build_exit=0, image_id=current_image, stage="complete", status="PASS")
            save()
            receipt = {"status": "PREPARED", "checked_at": stamp(),
                       "bundle_identity": plan["bundle_identity"], "image_identity": plan["image_identity"],
                       "image_id": current_image, "model_download_command_exit": 0,
                       "container_build_exit": 0, "runtime_started": False,
                       "source_files": plan["source_files"], "source_revision": plan["source_revision"]}
            atomic(deployment / "installed.json", receipt)
            return record
        except BaseException as error:
            record.update(status="FAILED", error_type=type(error).__name__)
            save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("run", "status"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.operation == "status":
        result = status(args.root)
    else:
        if os.getuid() == 0:
            raise ValueError("Preparation must run as the admitted unprivileged Brev account")

        def interrupted(signum, frame):
            raise InterruptedError("Supervised preparation was stopped")

        signal.signal(signal.SIGTERM, interrupted)
        result = prepare(args.root, read_record(args.root / "deployment/install-plan.json"))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error_type": type(error).__name__}), flush=True)
        raise SystemExit(1)
