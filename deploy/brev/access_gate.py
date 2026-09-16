#!/usr/bin/env python3
"""Prove controller control through the approved Tailscale peer and local wrapper."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent
STOP = HERE.parents[1] / "STOP"
DEFAULT_PROFILE = HERE / "profiles/brev-rtx6000.json"
TARGET = json.loads(DEFAULT_PROFILE.read_text())["network"]["tailscale_hostname"]
EXPECTED_GPU = json.loads(DEFAULT_PROFILE.read_text())["accelerator"]["name"]
ALIAS = TARGET + "-tailnet"
WRAPPER = Path(os.environ.get("PAAI_TRANSPORT_WRAPPER",
                             Path.home() / ".local/bin/totally-not-ssh.sh")).expanduser()
PRIVATE_CONFIG = Path.home() / ".ssh/config.d" / (TARGET + ".conf")
MAIN_CONFIG = Path.home() / ".ssh/config"
KEY = Path.home() / ".brev/brev.pem"


def target_config(hostname):
    if (not isinstance(hostname, str)
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", hostname)):
        raise ValueError("The profile needs one lowercase Tailscale hostname")
    return hostname, hostname + "-tailnet", Path.home() / ".ssh/config.d" / (hostname + ".conf")


def select_profile(profile):
    """Select transport before any probe or transfer; credentials stay local."""
    global TARGET, ALIAS, PRIVATE_CONFIG, EXPECTED_GPU
    network = profile.get("network")
    if not isinstance(network, dict) or network.get("access") != "tailscale":
        raise ValueError("The deployment profile requires Tailscale access")
    selected = target_config(network.get("tailscale_hostname"))
    TARGET, ALIAS, PRIVATE_CONFIG = selected
    EXPECTED_GPU = profile.get("accelerator", {}).get("name", EXPECTED_GPU)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run(argv, timeout=20):
    return subprocess.run(list(map(str, argv)), stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=timeout)


def validate_wrapper():
    if not WRAPPER.is_absolute() or not WRAPPER.is_file() or not os.access(WRAPPER, os.X_OK):
        raise ValueError("PAAI_TRANSPORT_WRAPPER must name an absolute executable file; "
                         "the default is ~/.local/bin/totally-not-ssh.sh")


def owned_regular(path):
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise ValueError(f"Expected a regular file owned by the current account: {path.name}")
    return metadata


def private_directory(path):
    if path.exists() or path.is_symlink():
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise ValueError("Private access configuration needs an owned directory")
    else:
        path.mkdir(mode=0o700)
    path.chmod(0o700)


def private_text(path, value):
    if path.exists() or path.is_symlink():
        metadata = owned_regular(path)
        if path.read_text() == value and stat.S_IMODE(metadata.st_mode) == 0o600:
            return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False,
                                         prefix=".paai-access-", encoding="utf-8") as output:
            temporary = Path(output.name)
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def configure(ip):
    address = ipaddress.ip_address(ip)
    if address not in ipaddress.ip_network("100.64.0.0/10"):
        raise ValueError("The target must have a Tailscale IPv4 address")
    if stat.S_IMODE(owned_regular(KEY).st_mode) != 0o600:
        raise ValueError("The controller's private Brev key must have mode 0600")
    for path in (PRIVATE_CONFIG, MAIN_CONFIG):
        if path.exists() or path.is_symlink():
            owned_regular(path)
    private_directory(PRIVATE_CONFIG.parent.parent)
    private_directory(PRIVATE_CONFIG.parent)
    text = (f"Host {ALIAS}\n    HostName {address}\n    User ubuntu\n"
            f"    IdentityFile {KEY}\n    IdentitiesOnly yes\n"
            "    BatchMode yes\n    ConnectTimeout 12\n    ConnectionAttempts 1\n"
            "    ServerAliveInterval 10\n    ServerAliveCountMax 2\n"
            "    RequestTTY no\n    StrictHostKeyChecking accept-new\n")
    private_text(PRIVATE_CONFIG, text)
    include = f"Include {PRIVATE_CONFIG}\n"
    existing = MAIN_CONFIG.read_text() if MAIN_CONFIG.exists() else ""
    if include not in existing:
        private_text(MAIN_CONFIG, include + existing)


def probe(profile=None):
    if STOP.exists():
        return {"status": "STOPPED", "checked_at": stamp()}
    if profile is not None:
        select_profile(profile)
    record = {"status": "WAITING_FOR_ACCESS", "checked_at": stamp(),
              "target": TARGET, "transport": "controller local wrapper over Tailscale",
              "remote_command_executed": False}
    try:
        result = run(["tailscale", "status", "--json"])
        if result.returncode:
            raise RuntimeError("controller Tailscale status failed; raw output withheld")
        status = json.loads(result.stdout)
        record["matrix_tailscale"] = {
            "state": status.get("BackendState"),
            "key_expiry": status.get("Self", {}).get("KeyExpiry"),
        }
        if status.get("BackendState") != "Running":
            raise RuntimeError("controller Tailscale is not in the Running state")
        matches = [peer for peer in status.get("Peer", {}).values()
                   if peer.get("HostName", "").lower() == TARGET
                   or peer.get("DNSName", "").split(".")[0].lower() == TARGET]
        if len(matches) != 1:
            record["reason"] = "Target is absent or ambiguous in The controller's Tailscale peer list"
        else:
            peer = matches[0]
            record["peer"] = {key: peer.get(key) for key in
                              ("HostName", "DNSName", "Online", "TailscaleIPs")}
            addresses = [ip for ip in peer.get("TailscaleIPs", []) if ":" not in ip]
            if len(addresses) != 1:
                raise ValueError("Target has no unique Tailscale IPv4 address")
            validate_wrapper()
            configure(addresses[0])
            if STOP.exists():
                return {"status": "STOPPED", "checked_at": stamp()}
            command = """set -eu
hostname
id
nvidia-smi
test "$(id -un)" = ubuntu
nvidia-smi --query-gpu=name --format=csv,noheader | /usr/bin/grep -Fx -- EXPECTED_GPU_NAME
sudo -n true
printf 'PAAI_ACCESS_GATE_PASSED\\n'
"""
            command = command.replace("EXPECTED_GPU_NAME", shlex.quote(EXPECTED_GPU))
            remote = run([WRAPPER, "-F", PRIVATE_CONFIG, ALIAS, command], timeout=40)
            record["remote_command_executed"] = True
            record["exit_code"] = remote.returncode
            # This fixed command prints only host, user, GPU and sudo evidence.
            record["stdout"] = remote.stdout[-20000:]
            record["stderr"] = remote.stderr[-2000:]
            if remote.returncode == 0 and "PAAI_ACCESS_GATE_PASSED" in remote.stdout:
                record["status"] = "PASS"
                record["alias"] = ALIAS
            else:
                record["reason"] = "The finite wrapper command did not pass all access checks"
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        record["reason"] = ("Finite access probe timed out" if isinstance(error, subprocess.TimeoutExpired)
                            else str(error))
    path = HERE / "access" / ("probe-" + record["checked_at"].replace(":", "-") + ".json")
    atomic(path, record)
    record["evidence"] = str(path.relative_to(HERE))
    atomic(HERE / "access/current.json", record)
    if record["status"] == "PASS":
        atomic(HERE / "access/PASSED.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--wait-seconds", type=int, default=0)
    parser.add_argument("--interval", type=int, default=45)
    args = parser.parse_args()
    select_profile(json.loads(args.profile.read_text()))
    deadline = time.monotonic() + min(max(args.wait_seconds, 0), 3600)
    while True:
        record = probe()
        print(json.dumps({key: record[key] for key in
                          ("status", "checked_at", "reason", "evidence") if key in record}), flush=True)
        if record["status"] == "PASS":
            return 0
        if record["status"] == "STOPPED":
            return 0
        if time.monotonic() >= deadline:
            return 75
        until = min(deadline, time.monotonic() + max(30, min(args.interval, 60)))
        while time.monotonic() < until and not STOP.exists():
            time.sleep(min(2, until - time.monotonic()))


if __name__ == "__main__":
    raise SystemExit(main())
