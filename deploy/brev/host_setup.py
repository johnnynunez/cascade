#!/usr/bin/env python3
"""Install an isolated Docker daemon whose entire data root is on the NVMe volume."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import tempfile

import preflight

UNIT = "paai-demo-docker.service"
UNIT_DIRECTORY = Path("/etc/systemd/system")
HOME_DIRECTORY = Path.home()
DOCKER_SOCKET = "unix:///run/paai-demo-docker.sock"
PATH_CHARACTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/_-."


def run(command, timeout=30):
    result = subprocess.run(list(map(str, command)), capture_output=True, text=True,
                            stdin=subprocess.DEVNULL, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{Path(str(command[0])).name} failed with exit {result.returncode}; output withheld")
    return result.stdout.strip()


def unit_text(root, dockerd, mount):
    # Paths have been constrained before reaching systemd's command syntax.
    return f"""[Unit]
Description=Physical Agentic AI dedicated Docker storage
After=network-online.target
Wants=network-online.target
RequiresMountsFor={root}
StartLimitIntervalSec=300
StartLimitBurst=3

[Service]
Type=notify
ExecStartPre=/usr/bin/mountpoint -q {mount}
ExecStart={dockerd} --config-file={root}/deployment/docker-daemon.json
Restart=on-failure
RestartSec=5
TimeoutStartSec=90
TimeoutStopSec=90
LimitNOFILE=infinity
Delegate=yes
KillMode=process

[Install]
WantedBy=multi-user.target
"""


def ensure_home_link(root):
    conflicts = []
    for name in ("paai-demo", "cascade-demo"):
        link = HOME_DIRECTORY / name
        if link.is_symlink() and link.resolve() == root:
            return {"home_link": str(link), "home_link_status": "reused", "home_link_conflicts": conflicts}
        if link.exists() or link.is_symlink():
            conflicts.append(str(link))
            continue
        try:
            link.symlink_to(root, target_is_directory=True)
        except FileExistsError:
            conflicts.append(str(link))
            continue
        return {"home_link": str(link), "home_link_status": "created", "home_link_conflicts": conflicts}
    return {"home_link": None, "home_link_status": "conflict", "home_link_conflicts": conflicts}


def install(profile):
    root = Path(profile["storage"]["root"])
    mount = Path(profile["storage"]["mount"]).resolve(strict=True)
    if (not root.is_absolute() or root == mount or not root.resolve().is_relative_to(mount)
            or any(c not in PATH_CHARACTERS for c in str(root))):
        raise ValueError("Invalid dedicated data root")
    if not os.path.ismount(mount) or mount.stat().st_dev == Path("/").stat().st_dev:
        raise ValueError("Data volume must be mounted separately from root")
    disk = preflight.validate_storage(profile, installed=True)
    root, mount = Path(disk["root"]), Path(disk["mount"])
    if any(c not in PATH_CHARACTERS for path in (root, mount) for c in str(path)):
        raise ValueError("Resolved data-volume paths are unsafe for systemd configuration")
    dockerd, nvidia = shutil.which("dockerd"), shutil.which("nvidia-container-runtime")
    if not dockerd or not nvidia:
        raise ValueError("Existing Docker and NVIDIA container runtime are required")
    if not Path(dockerd).is_absolute() or any(c not in PATH_CHARACTERS for c in dockerd):
        raise ValueError("Docker executable path is unsafe for systemd configuration")
    run(["sudo", "-n", "true"])
    if run(["id", "-un"]) != "ubuntu":
        raise ValueError("Host setup requires the selected ubuntu account")
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        raise ValueError("Host setup requires the unprivileged ubuntu account")
    if root.exists() and (not root.is_dir() or root.stat().st_uid != uid):
        raise ValueError("Existing deployment root must be owned by the ubuntu account")
    deployment = root / "deployment"
    if deployment.exists() or deployment.is_symlink():
        metadata = deployment.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != uid:
            raise ValueError("Existing deployment configuration directory must be owned and regular")
    config = {"data-root": str(root / "docker"), "exec-root": "/run/paai-demo-docker",
              "pidfile": "/run/paai-demo-docker.pid", "hosts": [DOCKER_SOCKET],
              "group": "docker", "bridge": "none", "iptables": False, "ip6tables": False,
              "ip-masq": False, "storage-driver": "overlay2",
              "runtimes": {"nvidia": {"path": nvidia, "runtimeArgs": []}},
              "log-driver": "local", "log-opts": {"max-size": "20m", "max-file": "3"}}
    config_path = deployment / "docker-daemon.json"
    expected = json.dumps(config, indent=2) + "\n"
    unit_path = UNIT_DIRECTORY / UNIT
    wanted_unit = unit_text(root, dockerd, mount)
    for path in (unit_path, config_path):
        if (path.exists() or path.is_symlink()) and not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError("Existing Docker unit and configuration must be regular files")
    if unit_path.exists() and unit_path.read_text() != wanted_unit:
        raise ValueError("An existing dedicated Docker unit differs; review it before replacement")
    if config_path.exists() and config_path.read_text() != expected:
        raise ValueError("An existing dedicated Docker config differs; review it before replacement")
    if not root.exists():
        run(["sudo", "-n", "install", "-d", "-m", "0750", "-o", str(uid), "-g", str(gid), root])
        if not root.is_dir() or root.stat().st_uid != uid or root.stat().st_dev != mount.stat().st_dev:
            raise ValueError("Dedicated deployment root creation did not preserve ownership and volume")
    deployment.mkdir(mode=0o700, exist_ok=True)
    if not config_path.exists():
        with tempfile.NamedTemporaryFile(mode="w", prefix=".docker-config-", dir=deployment,
                                         delete=False) as file:
            temporary = Path(file.name)
            file.write(expected)
            file.flush()
            os.fsync(file.fileno())
        try:
            temporary.replace(config_path)
        finally:
            temporary.unlink(missing_ok=True)
    if not unit_path.exists():
        with tempfile.NamedTemporaryFile(mode="w", prefix="paai-docker-", delete=False) as file:
            file.write(wanted_unit)
            temporary = Path(file.name)
        try:
            run(["sudo", "-n", "install", "-m", "0644", temporary, unit_path])
        finally:
            temporary.unlink(missing_ok=True)
        run(["sudo", "-n", "systemctl", "daemon-reload"])
    run(["sudo", "-n", "systemctl", "enable", "--now", UNIT], timeout=100)
    actual = run(["docker", "--host", DOCKER_SOCKET, "info", "--format", "{{.DockerRootDir}}"], timeout=30)
    if Path(actual).resolve() != (root / "docker").resolve():
        raise ValueError("Dedicated Docker storage verification failed")
    return {"status": "PASS", "socket": DOCKER_SOCKET, "data_root": actual,
            "uid": uid, "gid": gid, "mount": str(mount),
            **ensure_home_link(root),
            "existing_docker_daemon_modified": False, "driver_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = install(json.loads(args.profile.read_text()))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"status": "FAIL", "reason": str(error)}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
